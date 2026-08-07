import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .assignment_config import AttributeAssignmentConfig
from .data_sources import DataSourceManager
from .strategies import StrategyFactory, Refusal, AssignmentRefused
from may.utils.attribute_access import get_person_attribute
from may.parallel import parallel_map, PopulationSlice, GeoUnitSlice

logger = logging.getLogger("may.attribute_assignment.assigner")

# Set by the parent immediately before a parallel_map dispatch and read by the
# worker over fork copy-on-write, so the (large, read-only) world graph is never
# pickled into a Slice payload (ADR-0003). Holds the live assigner plus the
# id->person index the worker rebuilds its Slice's people from.
_ACTIVE_ASSIGNMENT_CONTEXT = None


class AttributeAssignmentError(Exception):
    """Raised when assignment can't complete on the given data/config. Mirrors
    PopulationError/VenueError/HouseholdError: every eligible person is assigned
    on complete data, or the build fails loud. Refusals are counted and raised
    on here, once per attribute with a total; an engine fault aborts where it
    happened, as itself, and never reaches this."""


def assign_attributes(venue_manager, config_path: str, geo_units: Optional[set] = None,
                      geo_units_by_level: Optional[Dict[str, set]] = None,
                      population=None, workers: Optional[int] = None,
                      max_tasks_per_child: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to assign attributes to a population.

    Args:
        venue_manager: VenueManager with households and venues
        config_path: Path to YAML configuration file
        geo_units: Optional set of geo unit codes to preload data for
        geo_units_by_level: The same names split by hierarchy level. Sources that
            declare the level their values live at match against one level only.
        population: Population manager; required for person-level assignment,
            which collects people from it rather than from venue membership
        workers: Process-pool size. None resolves to the SLURM allocation,
            then the machine's CPU count.
        max_tasks_per_child: Recycling cadence. None never retires a worker,
            matching today's behaviour.

    Returns:
        Assignment statistics dictionary
    """
    # Load configuration
    config = AttributeAssignmentConfig.from_yaml(config_path)

    # Initialize data manager
    data_manager = DataSourceManager(config)

    # Surface a data-source load failure as AttributeAssignmentError so
    # create_world exits cleanly.
    try:
        if geo_units:
            logger.info(f"Preloading data for {len(geo_units)} geographical units...")
            data_manager.load_all(geo_units, geo_units_by_level)
        else:
            logger.info("Loading all data sources...")
            data_manager.load_all()
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
        raise AttributeAssignmentError(
            f"could not load data sources for '{config_path}': {e}"
        ) from e

    # Create assigner and run
    assigner = AttributeAssigner(config, data_manager)
    stats = assigner.assign_all(venue_manager, population=population,
                               workers=workers,
                               max_tasks_per_child=max_tasks_per_child)

    return stats
class AttributeAssigner:
    """
    Main orchestrator for attribute assignment.

    Uses structure-based assignment with straightforward role logic.
    """

    def __init__(self, config: AttributeAssignmentConfig, data_manager: DataSourceManager):
        """
        Initialize attribute assigner.

        Args:
            config: Attribute assignment configuration
            data_manager: Data source manager with loaded data
        """
        self.config = config
        self.data_manager = data_manager
        self.attribute_name = config.attribute_name

        # Logging settings
        self.verbose = config.settings.get('logging', {}).get('detailed_assignment_logging', False)

        # Global seed for per-Slice RNG streams under parallel assignment. A
        # Slice's stream is derived from this plus the Slice's stable person-id
        # digest (ADR-0002), so output is reproducible across worker count.
        self._global_seed = config.settings.get('random_seed', 0) if config.settings else 0

        # Cache strategy objects to avoid repeated creation
        self._strategy_cache = {}  # Maps strategy config hash to strategy instance
        # Pre-compute filter configuration
        self._has_filters = hasattr(config, 'filters') and config.filters
        self._optimized_filters = []
        # Filters that inspect the person's activity_map (assigned venues/subsets).
        # Populated below.
        self._activity_venue_filters = []
        if self._has_filters:
            for name, cfg in config.filters.items():
                if name == 'activities':
                    continue
                # activity_venue: require an assigned venue of a given type
                # (optionally in a given subset) under some activity.
                if cfg.get('type') == 'activity_venue':
                    self._activity_venue_filters.append({
                        'activity': cfg.get('activity', 'primary_activity'),
                        'venue_types': set(cfg.get('venue_types', [])),
                        'subset_names': set(cfg.get('subset_names', [])),
                    })
                    continue
                attr = cfg.get('attribute')
                ftype = cfg.get('type')
                num = cfg.get('numerical', {})
                # Categorical "include" values: accept either nested
                # (categorical: {values: [...]}) or flat (values: [...]) form.
                cat_values = cfg.get('categorical', {}).get('values', cfg.get('values'))
                # Categorical "exclude" values: reject people whose value is in
                # this list (e.g. the out-of-boundary sentinel).
                cat_exclude = cfg.get('categorical', {}).get('exclude', cfg.get('exclude'))
                self._optimized_filters.append({
                    'attr': attr,
                    'type': ftype,
                    'min': num.get('min'),
                    'max': num.get('max'),
                    'values': cat_values,
                    'exclude': cat_exclude,
                })

        self._activity_filters = config.filters.get('activities', {}) if self._has_filters else {}
        self._include_activities = self._activity_filters.get('include', [])
        self._exclude_activities = self._activity_filters.get('exclude', [])

        # Statistics
        self.stats = self._fresh_stats()

    @staticmethod
    def _fresh_stats() -> Dict[str, Any]:
        """A zeroed statistics structure. Reused per-Slice in workers so each
        Slice reports only its own contribution, which the parent then merges
        (see _merge_assignment_stats)."""
        return {
            'total_people': 0,
            'people_in_households': 0,
            'people_in_other_residences': 0,
            'households_processed': 0,
            'other_residences_processed': 0,
            'assignments_by_venue_type': defaultdict(int),
            'assignments_by_rule': defaultdict(int),
            'assignments_by_role': defaultdict(int),
            'assignments_by_strategy': defaultdict(int),
            'attribute_distribution': defaultdict(int),
            'household_structure_counts': defaultdict(int),
            'refusals': 0,
            'filtered_people': 0,  # People filtered out by age/activity filters
            'assigned_people': 0,  # People successfully assigned
            # People dropped by a categorical 'exclude' filter, keyed by attribute
            # (e.g. the out-of-boundary sentinel).
            'value_excluded': defaultdict(int),
        }

    def _get_or_create_strategy(self, assignment_config):
        """
        Get cached strategy or create new one.

        Args:
            assignment_config: Strategy configuration dict (from AttributeAssignmentRule)

        Returns:
            Strategy instance
        """
        # Use object id as cache key

        config_key = id(assignment_config)

        # Check cache
        if config_key not in self._strategy_cache:
            # Create new strategy
            self._strategy_cache[config_key] = StrategyFactory.create_strategy(
                assignment_config, self.data_manager
            )

        return self._strategy_cache[config_key]

    def assign_all(self, venue_manager, population=None, workers=None,
                   max_tasks_per_child=None) -> Dict[str, Any]:
        """
        Assign attribute based on assignment level (household or person).

        Args:
            venue_manager: VenueManager with households and people
            population: Population manager; required for person-level
                assignment (the population is read directly, not rebuilt
                from venue membership, so gutted subsets stay untouched)
            workers: Process-pool size for the geo_unit-partitioned parallel
                assignment. None resolves to the SLURM/CPU default; 1 runs
                in-process (no fork). Output is identical across worker count.
            max_tasks_per_child: Recycling cadence for the same pool. None
                never retires a worker, matching today's behaviour.

        Returns:
            Dictionary with assignment statistics
        """
        logger.info(f"Starting attribute assignment for '{self.attribute_name}'...")
        logger.info(f"Assignment level: {self.config.assignment_level}")
        logger.info("=" * 80)

        # Branch based on assignment level
        if self.config.assignment_level == "person":
            if population is None:
                raise AttributeAssignmentError(
                    f"person-level assignment for '{self.attribute_name}' needs a "
                    f"population manager — people are collected from the population, "
                    f"not from venue membership"
                )
            self._assign_all_people(population, workers=workers,
                                    max_tasks_per_child=max_tasks_per_child)
        elif self.config.assignment_level == "person_by_residence":
            self._assign_all_residences(venue_manager, workers=workers,
                                        max_tasks_per_child=max_tasks_per_child)
        else:
            raise ValueError(f"Unknown assignment_level: '{self.config.assignment_level}'. "
                           f"Expected 'person' or 'person_by_residence'.")

        # Report statistics
        self._report_statistics()

        # Fail loud on the total. Each refusal carried its own reason and was
        # logged with it as it happened, so this counts rather than guesses.
        if self.stats['refusals'] > 0:
            raise AttributeAssignmentError(
                f"'{self.attribute_name}': {self.stats['refusals']} refusal(s). "
                f"Each was logged above with the person and the reason given. "
                f"The engine assigns every eligible person on complete data or "
                f"fails loud."
            )

        return self.stats

    def _assign_all_residences(self, venue_manager, workers=None,
                               max_tasks_per_child=None):
        """Assign attributes at residence level (households and communal
        establishments).

        Venues are partitioned into geo_unit Slices — a residence sits wholly in
        one geo_unit, so each household's members and role logic stay within a
        single Slice. Each worker runs its geo_unit's residences and returns
        compact (person-id, value) arrays the parent applies (ADR-0003)."""
        all_venues = venue_manager.get_all_venues_list()
        logger.info(f"Found {len(all_venues)} total venues")

        # Count total people across ALL venues for accurate statistics
        total_people_in_simulation = sum(venue.size() for venue in all_venues)

        # Get the residence venue types assigned by household structure
        # (defaults to ["household"]); other residences use venue_assignment_rules.
        residence_venue_types = self.config.residence_venue_types or ["household"]

        # Partition into geo_unit Slices and dispatch.
        slices, geo_to_venues, id_to_person = _partition_venues_by_geo(all_venues)
        logger.info(f"  Residences across {len(slices)} geo_unit(s)")
        results = self._dispatch_residence_slices(
            slices, geo_to_venues, id_to_person, residence_venue_types, workers,
            max_tasks_per_child)
        self._apply_assignment_results(results, id_to_person)

        # Set total_people from the authoritative venue count
        self.stats['total_people'] = total_people_in_simulation
        logger.info(f"✓ Processed {self.stats['households_processed']} households")
        if self.stats['other_residences_processed']:
            logger.info(f"✓ Assigned {self.stats['people_in_other_residences']} "
                        f"people in other residences")

    def _dispatch_residence_slices(self, slices, geo_to_venues, id_to_person,
                                   residence_venue_types, workers,
                                   max_tasks_per_child=None):
        """Publish the world to workers via the module-global context (read over
        fork copy-on-write, never pickled), run each geo_unit Slice, then clear
        it."""
        global _ACTIVE_ASSIGNMENT_CONTEXT
        _ACTIVE_ASSIGNMENT_CONTEXT = _AssignmentContext(
            assigner=self, id_to_person=id_to_person,
            geo_to_venues=geo_to_venues, residence_venue_types=residence_venue_types)
        try:
            return parallel_map(_assign_residence_slice, slices,
                                global_seed=self._global_seed, workers=workers,
                                max_tasks_per_child=max_tasks_per_child)
        finally:
            _ACTIVE_ASSIGNMENT_CONTEXT = None

    def _record_refusal(self, refusal):
        """Count and log one Refusal.

        The single counting path for both shapes a Refusal arrives in: raised by
        `assign` inside an `AssignmentRefused`, returned by `assign_batch` as a
        value. Batch and sequential therefore report identical totals on
        identical input by construction. See
        docs/adr/0008-a-refusal-is-a-per-person-result-an-engine-fault-is-an-abort.md
        """
        self.stats['refusals'] += 1
        logger.warning(str(refusal))

    def _refuse_person(self, person, reason: str):
        """The engine's own verdict on one person, for the refusals it reaches
        before any strategy exists: no rule for the venue type, no classifiable
        household structure, no role, no rule for that (structure, role).

        There is no strategy to attribute these to — that absence is precisely
        what is being refused — so the attribute names the decider instead. They
        are counted and logged per person like any other Refusal, so the
        end-of-attribute raise can send the operator to a cause for each.
        """
        self._record_refusal(Refusal(
            person_id=person.id, reason=reason,
            refused_by=f"Assignment of '{self.attribute_name}'"))

    def _assign_other_residences(self, venues):
        """
        Assign attributes to people in non-household residences (care homes, dorms, etc.).
        Uses rules defined in venue_assignment_rules section of config.

        Args:
            venues: List of non-household residence venues

        Returns:
            Number of people assigned
        """
        people_assigned = 0
        venues_processed = 0

        for venue in venues:
            members = venue.get_all_members()
            if not members:
                continue

            # Find the assignment rule for this venue type
            venue_rule = None
            for rule in self.config.venue_assignment_rules:
                if venue.type in rule.get('venue_types', []):
                    venue_rule = rule
                    break

            # An attribute's scope belongs in its eligibility filters, not in
            # which venue types happen to have had a rule written for them.
            if not venue_rule:
                logger.warning(f"No assignment rule for venue type '{venue.type}': "
                               f"refusing its {len(members)} members")
                self.stats['refusals'] += len(members)
                continue

            # Get the assignment strategy from the rule. Outside the per-person
            # try: a strategy that cannot be constructed is an engine fault, and
            # no member of this venue could ever be assigned, so there is no
            # per-person verdict to give.
            assignment_config = venue_rule.get('assignment', {})
            strategy = self._get_or_create_strategy(assignment_config)
            venue_assigned = 0

            # Assign each person
            for person in members:
                # Skip if already assigned
                if self.attribute_name in person.properties:
                    continue

                # Create context for strategy
                context = {
                    'attribute_name': self.attribute_name,
                    'venue_type': venue.type
                }

                # Per person, not per venue: one refused member must not cost
                # everybody queued behind them their assignment.
                try:
                    value = strategy.assign(person, venue, context)
                except AssignmentRefused as refused:
                    self._record_refusal(refused.refusal)
                    continue

                person.properties[self.attribute_name] = value
                self.stats['attribute_distribution'][value] += 1
                self.stats['assignments_by_strategy'][f'venue_{venue.type}'] += 1
                self.stats['assignments_by_venue_type'][venue.type] += 1
                people_assigned += 1
                venue_assigned += 1

            if venue_assigned > 0:
                venues_processed += 1

        # Update stats
        self.stats['people_in_other_residences'] = people_assigned
        self.stats['other_residences_processed'] = venues_processed

        return people_assigned

    def _passes_filters(self, person):
        """
        Check if person passes all configured filters.

        Args:
            person: Person object

        Returns:
            bool: True if person passes all filters
        """
        if not self._has_filters:
            return True

        # 1. Attribute filters
        for f in self._optimized_filters:
            person_value = get_person_attribute(person, f['attr'])

            if person_value is None:
                continue

            if f['type'] == 'numerical':
                vmin = f['min']
                if vmin is not None and person_value < vmin:
                    return False
                vmax = f['max']
                if vmax is not None and person_value > vmax:
                    return False
            elif f['type'] == 'categorical':
                values = f['values']
                if values is not None and person_value not in values:
                    return False
                exclude = f['exclude']
                if exclude is not None and person_value in exclude:
                    self.stats['value_excluded'][f['attr']] += 1
                    return False

        # 2. Activity filters (Fast set intersection check)
        if self._include_activities or self._exclude_activities:
            # use direct attribute access for activities (it's a slot)
            person_activities = person.activities
            
            if self._include_activities:
                # simple loop is faster than generator for small lists
                has_activity = False
                for a in self._include_activities:
                    if a in person_activities:
                        has_activity = True
                        break
                if not has_activity:
                    return False

            if self._exclude_activities:
                for a in self._exclude_activities:
                    if a in person_activities:
                        return False

        # 2b. Activity-venue filters: require a venue of a given type (optionally
        # in a given subset) already assigned under some activity, so commute is
        # gated on people who hold a workplace venue.
        for f in self._activity_venue_filters:
            venue_map = person.activity_map.get(f['activity'], {})
            venue_types = f['venue_types']
            subset_names = f['subset_names']
            matched = False
            for vt, subsets in venue_map.items():
                if venue_types and vt not in venue_types:
                    continue
                if subset_names:
                    if any(getattr(s, 'subset_name', None) in subset_names for s in subsets):
                        matched = True
                        break
                elif subsets:
                    matched = True
                    break
            if not matched:
                return False

        return True

    def _assign_all_people(self, population, workers=None, max_tasks_per_child=None):
        """
        Assign attributes at person level.

        People come straight from the population manager: no venue-membership
        iteration, so each person is seen exactly once and gutted subsets
        (flushed household membership) never block assignment.

        Eligible people are partitioned into geo_unit-keyed population Slices and
        assigned across a bounded process pool; each worker returns compact
        (person-id, value) arrays which the parent applies (ADR-0003).
        """
        all_people = population.get_all_people()

        logger.info(f"Found {len(all_people)} total people")
        logger.info("")

        # Check required attributes
        self._check_required_attributes(all_people)

        # Pre-filter all people once
        logger.info("Pre-filtering people by age/activity filters...")
        eligible_people = []
        for person in all_people:
            if self._passes_filters(person):
                eligible_people.append(person)
            else:
                self.stats['filtered_people'] += 1

        self.stats['total_people'] = len(all_people)
        logger.info(f"  ✓ Eligible for assignment: {len(eligible_people)} / {len(all_people)} people")
        logger.info(f"  ✓ Filtered out: {self.stats['filtered_people']} people")
        for attr, n in self.stats['value_excluded'].items():
            logger.info(
                f"  ✓ Excluded {n} people from '{self.attribute_name}' "
                f"({attr} value in exclude list, e.g. out of boundary)"
            )
        logger.info("")

        # Get assignment rule and strategy
        rule = self.config.get_person_assignment_rule()
        if not rule:
            logger.warning(f"No assignment rule for person-level attribute '{self.attribute_name}'")
            self.stats['refusals'] = len(eligible_people)
            logger.info("")
            return

        # Capture parent-computed pre-filter counters before dispatch: the
        # in-process (single-worker) path shares this assigner and resets
        # self.stats per Slice, so they cannot be read back afterwards.
        preserved = {
            'filtered_people': self.stats['filtered_people'],
            'total_people': self.stats['total_people'],
            'value_excluded': self.stats['value_excluded'],
        }

        # Partition into geo_unit-keyed population Slices and dispatch.
        slices, id_to_person = _partition_people_by_geo(eligible_people)
        results = self._dispatch_person_slices(slices, id_to_person, rule, workers,
                                               max_tasks_per_child)
        self._apply_assignment_results(results, id_to_person, preserved)

        logger.info(f"✓ Processed {len(all_people)} people")
        logger.info(f"✓ Filtered {self.stats['filtered_people']} people (age/activity filters)")
        logger.info(f"✓ Assigned {self.stats['assigned_people']} people")
        logger.info(f"✓ Refused {self.stats['refusals']} people")
        logger.info("")

    def _dispatch_person_slices(self, slices, id_to_person, rule, workers,
                                max_tasks_per_child=None):
        """Publish the world to workers via the module-global context (read over
        fork copy-on-write, never pickled), run each Slice, then clear it."""
        global _ACTIVE_ASSIGNMENT_CONTEXT
        _ACTIVE_ASSIGNMENT_CONTEXT = _AssignmentContext(
            assigner=self, id_to_person=id_to_person, rule=rule)
        try:
            return parallel_map(_assign_person_slice, slices,
                                global_seed=self._global_seed, workers=workers,
                                max_tasks_per_child=max_tasks_per_child)
        finally:
            _ACTIVE_ASSIGNMENT_CONTEXT = None

    def _apply_assignment_results(self, results, id_to_person, preserved=None):
        """Apply each Slice's returned values to the parent's Person objects and
        merge the per-Slice stats. `preserved` carries any parent-computed
        counters (e.g. filtered/total/value_excluded) that must survive the
        merge unchanged."""
        merged = self._fresh_stats()
        for ids, values, extras, slice_stats in results:
            for person_id, value in zip(ids.tolist(), values.tolist()):
                if value is not None:
                    id_to_person[person_id].properties[self.attribute_name] = value
            for person_id, extra in extras.items():
                id_to_person[person_id].properties.update(extra)
            _merge_assignment_stats(merged, slice_stats)

        if preserved:
            merged.update(preserved)
        self.stats = merged

    def _assign_all_people_batch(self, eligible_people, strategy):
        """
        Batch assignment mode.

        Uses strategy's assign_batch method to process all people together.

        Args:
            eligible_people: List of pre-filtered people
            strategy: Assignment strategy with assign_batch method
        """
        logger.info("Processing people in batch mode...")
        total = len(eligible_people)

        # Progress tracking
        progress_interval = max(1, total // 20)  # Report every 5%

        # Prepare batch data
        logger.info(f"  Preparing batch data for {total:,} people...")
        households = [self._get_person_residence_venue(p) for p in eligible_people]
        contexts = [{'attribute_name': self.attribute_name}
                    for _ in eligible_people]

        # One verdict per person comes back; anything raised out of here is an
        # engine fault and aborts unwrapped.
        logger.info(f"  Running batch assignment...")
        results = strategy.assign_batch(eligible_people, households, contexts)

        # Assign results to people
        logger.info(f"  Applying results to people...")
        for i, (person, value) in enumerate(zip(eligible_people, results)):
            if isinstance(value, Refusal):
                self._record_refusal(value)
            else:
                # Handle single value or multiple values (dict)
                if isinstance(value, dict):
                    # Multiple attributes returned
                    for attr_name, attr_value in value.items():
                        person.properties[attr_name] = attr_value
                        if attr_name == self.attribute_name:
                            self.stats['attribute_distribution'][str(attr_value)] += 1
                else:
                    # Single attribute
                    person.properties[self.attribute_name] = value
                    self.stats['attribute_distribution'][str(value)] += 1

                self.stats['assignments_by_strategy'][strategy.strategy_type] += 1
                self.stats['assigned_people'] += 1

            # Log progress
            if (i + 1) % progress_interval == 0 or (i + 1) == total:
                progress = ((i + 1) / total) * 100
                logger.info(f"    Progress: {i+1:,}/{total:,} ({progress:.1f}%)")

        logger.info(f"✓ Batch processed {total:,} people")

    def _assign_all_people_sequential(self, eligible_people, strategy):
        """
        Standard sequential assignment mode.

        Args:
            eligible_people: List of pre-filtered people
            strategy: Assignment strategy (already created, reuse it!)
        """
        logger.info("Processing people sequentially...")
        total = len(eligible_people)

        # Progress tracking
        progress_interval = max(1, total // 20)  # Report every 5%

        # Sample tracking for debugging
        sample_size = min(10, total) if total > 0 else 0
        sample_indices = set(np.random.choice(total, sample_size, replace=False)) if total > 0 else set()
        samples_logged = []

        # Process each person with the pre-created strategy
        for i, person in enumerate(eligible_people):
            # Track if this is a sample person
            is_sample = i in sample_indices

            if is_sample:
                logger.debug(f"\n  [SAMPLE {len(samples_logged)+1}] Person {person.id}:")
                logger.debug(f"    Age: {person.age}, Sex: {person.sex}")
                logger.debug(f"    Geo Unit: {person.geographical_unit.name if person.geographical_unit else 'None'}")
                logger.debug(f"    Existing attributes: {list(person.properties.keys())}")

            # Reuse the pre-created strategy
            household = self._get_person_residence_venue(person)
            context = {'attribute_name': self.attribute_name, 'debug': is_sample}

            # Only a Refusal is caught: anything else is an engine fault and
            # must not be absorbed into a counter.
            try:
                value = strategy.assign(person, household, context)
            except AssignmentRefused as refused:
                self._record_refusal(refused.refusal)
            else:
                # Handle single value or multiple values (dict)
                if isinstance(value, dict):
                    for attr_name, attr_value in value.items():
                        person.properties[attr_name] = attr_value
                        if attr_name == self.attribute_name:
                            self.stats['attribute_distribution'][str(attr_value)] += 1
                else:
                    person.properties[self.attribute_name] = value
                    self.stats['attribute_distribution'][str(value)] += 1

                self.stats['assignments_by_strategy'][strategy.strategy_type] += 1
                self.stats['assigned_people'] += 1

            if is_sample:
                result = person.properties.get(self.attribute_name, "NOT_ASSIGNED")
                logger.debug(f"    Result: {self.attribute_name} = {result}")
                samples_logged.append((person.id, result))

            # Log progress
            if (i + 1) % progress_interval == 0 or (i + 1) == total:
                progress = ((i + 1) / total) * 100
                logger.info(f"  Progress: {i+1:,}/{total:,} ({progress:.1f}%)")

    def _assign_household(self, household):
        """
        Assign attribute to all people in a household.

        Main assignment flow:
        1. Classify household structure
        2. Sort people by configured assignment order
        3. For each person:
           a. Determine role based on subset + already assigned roles
           b. Get assignment rule for (structure, role)
           c. Execute strategy
           d. Track assigned roles

        Args:
            household: Venue object (type="household")
        """
        # Get all members
        members = household.get_all_members()
        if not members:
            return

        if self.verbose:
            logger.debug(f"\n{'=' * 80}")
            logger.debug(f"Processing Household {household.id} "
                        f"(geo_unit={household.geographical_unit.name if household.geographical_unit else 'None'})")
            logger.debug(f"  Members: {len(members)}")
            logger.debug(f"  Original pattern: {household.properties.get('original_pattern', 'N/A')}")
            logger.debug(f"  Allocation pattern: {household.properties.get('allocation_pattern', 'N/A')}")

        # Pre-calculate person categories (subsets) to avoid repeated lookups
        # UNIFIED STRUCTURE: activity_map['residence']['household'] = [subsets]
        person_categories = {}
        for person in members:
            category = "unknown"
            if "residence" in person.activity_map and "household" in person.activity_map["residence"]:
                res_subsets = person.activity_map["residence"]["household"]
                if res_subsets:
                    category = res_subsets[0].subset_name
            person_categories[person.id] = category

        # 1. Classify household structure
        # Pass pre-calculated categories if possible, but get_household_structure currently uses internal logic
        structure = self.config.get_household_structure(household, verbose=self.verbose)
        if not structure:
            if self.verbose:
                logger.debug(f"  Could not classify household {household.id}, skipping")
            else:
                logger.warning(f"Could not classify household {household.id}, skipping")
            self.stats['refusals'] += len(members)
            return

        # Store structure in household properties
        household.properties['_structure'] = structure
        self.stats['household_structure_counts'][structure] += 1

        if not self.verbose:
            logger.debug(f"Household {household.id}: structure={structure}, members={len(members)}")

        # Initialize assignment context
        context = {
            'attribute_name': self.attribute_name,
            'household_structure': structure,
        }

        # Track assigned roles (as a list to maintain order and count)
        assigned_roles: List[str] = []

        # 2. Get dependency-aware assignment order
        # This ensures that roles with 'inherit_from' are processed after their dependencies
        sorted_members = self._get_dependency_aware_order(
            members, structure, person_categories
        )

        # 3. Assign each person in order
        for person in sorted_members:
            category = person_categories.get(person.id, "unknown")
            
            if self.verbose:
                logger.debug(f"\n  Assigning {person} (category={category}):")

            # 3a. Determine role
            # Pass pre-calculated category
            role = self.config.get_person_role(
                person, structure, assigned_roles, verbose=self.verbose,
                person_category=category
            )

            if not role:
                if self.verbose:
                    logger.debug(f"    Could not determine role, skipping")
                else:
                    logger.warning(f"  Could not determine role for {person} in {household.id}")
                self.stats['refusals'] += 1
                continue

            # Track assigned roles
            assigned_roles.append(role)

            # Store person by role in context (for strategies to reference)
            person_key = f"{role}_person"
            context[person_key] = person

            # 3b. Get assignment rule
            # get_assignment_rule is already fairly fast, but could be memoized in config
            rule = self.config.get_assignment_rule(structure, role, verbose=self.verbose)

            if not rule:
                if self.verbose:
                    logger.debug(f"    No rule found for role '{role}', skipping")
                else:
                    logger.warning(f"  No rule for role '{role}' in structure '{structure}' for {person}")
                self.stats['refusals'] += 1
                continue

            # 3c. Create and execute strategy. Only a Refusal is caught:
            # anything else is an engine fault and aborts unwrapped.
            strategy = self._get_or_create_strategy(rule.assignment)
            try:
                value = strategy.assign(person, household, context)
            except AssignmentRefused as refused:
                self._record_refusal(refused.refusal)
            else:
                # Assign attribute to person's properties dict
                person.properties[self.attribute_name] = value

                # Update statistics
                self.stats['assignments_by_role'][role] += 1
                self.stats['assignments_by_strategy'][strategy.strategy_type] += 1
                self.stats['assigned_people'] += 1

                # Track distribution, using str() for unhashable types (e.g. lists)
                dist_key = str(value) if isinstance(value, (list, dict)) else value
                self.stats['attribute_distribution'][dist_key] += 1

                if self.verbose:
                    logger.debug(f"    ✓ Assigned: {self.attribute_name}={value} "
                               f"(role={role}, strategy={strategy.strategy_type})")
                else:
                    logger.debug(f"  {person}: {self.attribute_name}={value} (role={role})")

        if self.verbose:
            logger.debug(f"{'=' * 80}\n")

        self.stats['households_processed'] += 1
        self.stats['people_in_households'] += len(members)
        self.stats['total_people'] += len(members)

    def _get_dependency_aware_order(self, members, structure: str,
                                    person_categories: Dict[int, str] = None) -> List:
        """
        Get person assignment order that satisfies role dependencies.

        Uses a topological sort so inheritors are processed after the roles they
        inherit from. Members are ordered by id as the stable base/tie-breaker, so
        which role a person takes (primary vs secondary, child, elder) depends on
        their subset and this order, not on any configurable category priority.
        """
        # 1. Determine roles for everyone first (predictive)
        # We need to know who is who to build the dependency graph
        temp_assigned_roles = []
        person_to_role = {}

        # Stable base order (by id) used to predict roles, since role depends on order.
        base_sorted = sorted(members, key=lambda p: p.id)

        for person in base_sorted:
            category = person_categories.get(person.id, "unknown") if person_categories else "unknown"
            role = self.config.get_person_role(
                person, structure, temp_assigned_roles, verbose=False,
                person_category=category
            )
            if role:
                person_to_role[person.id] = role
                temp_assigned_roles.append(role)

        # 2. Build dependency graph
        # graph[A] = [B, C] means A must come before B and C
        adj = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Map roles to people
        role_to_people = defaultdict(list)
        for pid, role in person_to_role.items():
            role_to_people[role].append(pid)
            
        # Add edges based on assignment rule dependencies
        for pid, role in person_to_role.items():
            rule = self.config.get_assignment_rule(structure, role)
            if rule and rule.dependencies:
                for dep_role in rule.dependencies:
                    # If dep_role is in the household, add edges from all people with that role
                    if dep_role in role_to_people:
                        for dep_pid in role_to_people[dep_role]:
                            if dep_pid != pid: # Avoid self-dependency
                                adj[dep_pid].append(pid)
                                in_degree[pid] += 1

        # 3. Topological Sort (Kahn's Algorithm), tie-broken by base (id) order.
        queue = []
        # Initial nodes with no dependencies
        for p in base_sorted:
            if in_degree[p.id] == 0:
                queue.append(p)

        result = []
        processed_count = 0

        while queue:
            # Among equal dependency levels, keep the stable base (id) order.
            queue.sort(key=lambda p: [x.id for x in base_sorted].index(p.id))
            
            curr = queue.pop(0)
            result.append(curr)
            processed_count += 1
            
            for neighbor_id in adj[curr.id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    # Find person object for this ID
                    neighbor = next(p for p in members if p.id == neighbor_id)
                    queue.append(neighbor)

        # 4. Every member must be orderable. Cross-role cycles are rejected at
        #    config load, so unprocessed people here mean the derived per-person
        #    graph is inconsistent, so fail loud.
        if processed_count < len(members):
            processed_ids = {p.id for p in result}
            unordered = [p.id for p in base_sorted if p.id not in processed_ids]
            raise RuntimeError(
                f"Assignment ordering for structure {structure!r} left people "
                f"{unordered} unordered — a cross-role dependency cycle escaped "
                f"load-time validation."
            )

        return result

    def _get_person_category(self, person) -> str:
        """
        Get person's category (subset name) from their household activity.

        Args:
            person: Person object

        Returns:
            Category name or "unknown"
        """
        # UNIFIED STRUCTURE: activity_map['residence']['household'] = [subsets]
        if "residence" in person.activity_map and "household" in person.activity_map["residence"] and person.activity_map["residence"]["household"]:
            return person.activity_map["residence"]["household"][0].subset_name
        return "unknown"



    def _get_person_residence_venue(self, person):
        """Get residence venue for a person (e.g., household, pub, care home)."""
        # UNIFIED STRUCTURE: activity_map['residence'][venue_type] = [subsets]
        if "residence" in person.activity_map:
            for venue_type, subsets in person.activity_map["residence"].items():
                if subsets:
                    venue = subsets[0].venue
                    if venue is not None:
                        logger.debug(f"  Person {person.id} residence found: {venue_type} (ID={venue.id})")
                        return venue
        return None

    def _check_required_attributes(self, people):
        """Check and log required attribute availability."""
        if not self.config.required_attributes:
            return

        logger.info("Checking required attributes...")
        for attr_name, attr_config in self.config.required_attributes.items():
            if not attr_config.get('required', False):
                continue

            missing_count = sum(1 for p in people if attr_name not in p.properties)
            total_count = len(people)
            present_count = total_count - missing_count

            logger.info(f"  '{attr_name}': {present_count}/{total_count} people have this attribute")

            if missing_count > 0:
                logger.warning(f"    {missing_count} people missing required attribute '{attr_name}'")

        logger.info("")

    def _report_statistics(self):
        """Report assignment statistics."""
        logger.info("=" * 80)
        logger.info("ASSIGNMENT STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total people: {self.stats['total_people']}")

        # Show filtered/assigned/unassigned breakdown
        if self.stats.get('filtered_people', 0) > 0:
            logger.info(f"Filtered people (age/activity): {self.stats['filtered_people']}")
        if self.stats.get('assigned_people', 0) > 0:
            logger.info(f"Assigned people: {self.stats['assigned_people']}")
        if self.stats['refusals'] > 0:
            logger.info(f"Refusals: {self.stats['refusals']}")

        # Household-specific stats
        if self.stats['people_in_households'] > 0:
            logger.info(f"  In households: {self.stats['people_in_households']}")
        if self.stats['people_in_other_residences'] > 0:
            logger.info(f"  In other residences: {self.stats['people_in_other_residences']}")
        if self.stats['households_processed'] > 0:
            logger.info(f"Households processed: {self.stats['households_processed']}")
        if self.stats['other_residences_processed'] > 0:
            logger.info(f"Other residences processed: {self.stats['other_residences_processed']}")
        logger.info("")

        # Show breakdown by venue type if applicable
        if self.stats['assignments_by_venue_type']:
            logger.info("Assignments by venue type:")
            for venue_type, count in sorted(self.stats['assignments_by_venue_type'].items()):
                logger.info(f"  {venue_type}: {count}")

        # Household structure distribution (only if household-level)
        if self.stats['household_structure_counts']:
            logger.info("Household structures:")
            for structure, count in sorted(self.stats['household_structure_counts'].items()):
                logger.info(f"  {structure}: {count}")
            logger.info("")

        # Role distribution (only if household-level)
        if self.stats['assignments_by_role']:
            logger.info("Assignments by role:")
            for role, count in sorted(self.stats['assignments_by_role'].items()):
                logger.info(f"  {role}: {count}")
            logger.info("")

        # Strategy distribution
        if self.stats['assignments_by_strategy']:
            logger.info("Assignments by strategy:")
            for strategy, count in sorted(self.stats['assignments_by_strategy'].items()):
                logger.info(f"  {strategy}: {count}")
            logger.info("")

        # Attribute distribution (can be disabled via settings)
        show_distribution = self.config.settings.get('logging', {}).get('show_attribute_distribution', True)

        if show_distribution:
            logger.info(f"{self.attribute_name.capitalize()} distribution:")
            total_assigned = sum(self.stats['attribute_distribution'].values())
            for value, count in sorted(self.stats['attribute_distribution'].items()):
                percentage = (count / total_assigned * 100) if total_assigned > 0 else 0
                logger.info(f"  {value}: {count:6d} ({percentage:5.2f}%)")
            logger.info("")
        else:
            # Still show summary count even when distribution is hidden
            unique_values = len(self.stats['attribute_distribution'])
            total_assigned = sum(self.stats['attribute_distribution'].values())
            logger.info(f"{self.attribute_name.capitalize()} distribution: {unique_values} unique values, {total_assigned} total assignments")
            logger.info("")

        logger.info("=" * 80)

class _AssignmentContext:
    """Read-only handle the parent hands to workers over fork copy-on-write.
    Holds the live assigner (config, data_manager, strategy cache), the
    id->person index a Slice applies its values through, and the axis-specific
    payload: `rule` for the person axis; `geo_to_venues` +
    `residence_venue_types` for the residence axis."""
    __slots__ = ('assigner', 'id_to_person', 'rule',
                 'geo_to_venues', 'residence_venue_types')

    def __init__(self, assigner, id_to_person, rule=None,
                 geo_to_venues=None, residence_venue_types=None):
        self.assigner = assigner
        self.id_to_person = id_to_person
        self.rule = rule
        self.geo_to_venues = geo_to_venues
        self.residence_venue_types = residence_venue_types


def _partition_people_by_geo(people):
    """Partition people into geo_unit-keyed population Slices. A household sits
    in one geo_unit, so household-keyed strategies stay whole within a Slice.
    Slices and their person-ids are sorted for a stable dispatch order; each
    Slice's RNG stream derives from its sorted person-ids (ADR-0002)."""
    buckets = defaultdict(list)
    for person in people:
        geo_unit = getattr(getattr(person, 'geographical_unit', None),
                           'name', None) or '__no_geo__'
        buckets[geo_unit].append(person)

    slices = []
    id_to_person = {}
    for geo_unit in sorted(buckets):
        person_ids = sorted(person.id for person in buckets[geo_unit])
        slices.append(PopulationSlice(person_ids))
        for person in buckets[geo_unit]:
            id_to_person[person.id] = person
    return slices, id_to_person


def _assign_person_slice(population_slice, rng):
    """Worker: assign one population Slice and return compact arrays for the
    parent to apply. Seeds process-global np.random from the Slice's identity
    (strategies draw from global np.random), so the stream is fixed by the
    Slice, not by which worker runs it (ADR-0002)."""
    context = _ACTIVE_ASSIGNMENT_CONTEXT
    assigner = context.assigner
    np.random.seed(int(rng.integers(0, 2 ** 32)))

    people = [context.id_to_person[person_id]
              for person_id in population_slice.person_ids]
    # Isolate this Slice's stats; the parent merges each Slice's contribution.
    assigner.stats = assigner._fresh_stats()
    keys_before = [set(person.properties) for person in people]

    strategy = assigner._get_or_create_strategy(context.rule.assignment)
    if hasattr(strategy, 'assign_batch') and callable(getattr(strategy, 'assign_batch')):
        assigner._assign_all_people_batch(people, strategy)
    else:
        assigner._assign_all_people_sequential(people, strategy)

    return _collect_slice_result(assigner, people, keys_before)


def _collect_slice_result(assigner, people, keys_before):
    """Build the (ids, values, extras, stats) return for a processed Slice.

    `values` is the primary attribute per person shipped as one array (a raw
    unicode buffer for the common categorical case, object otherwise — no
    per-person tuple pickling). `extras` carries any additional keys written by
    multi-attribute strategies, keyed by person-id."""
    attribute_name = assigner.attribute_name
    person_ids = np.fromiter((person.id for person in people),
                             dtype=np.int64, count=len(people))
    assigned = [person.properties.get(attribute_name) for person in people]
    try:
        # Inferred dtype: a raw unicode/numeric buffer for the common
        # categorical or scalar attribute — no per-person object pickling.
        values = np.array(assigned)
    except ValueError:
        # Ragged (list-valued strategies) — fall back to an object array.
        values = np.array(assigned, dtype=object)

    extras = {}
    for person, previous_keys in zip(people, keys_before):
        added = set(person.properties) - previous_keys - {attribute_name}
        if added:
            extras[person.id] = {key: person.properties[key] for key in added}

    return person_ids, values, extras, _plain_stats(assigner.stats)


def _plain_stats(stats):
    """Convert nested defaultdicts to plain dicts for clean cross-process
    transport."""
    return {key: (dict(value) if isinstance(value, dict) else value)
            for key, value in stats.items()}


def _merge_assignment_stats(accumulated, slice_stats):
    """Fold one Slice's assignment stats into the running totals: counts add,
    nested count dicts add per key."""
    for key, value in slice_stats.items():
        if isinstance(value, dict):
            nested = accumulated.setdefault(key, defaultdict(int))
            for nested_key, nested_value in value.items():
                nested[nested_key] = nested.get(nested_key, 0) + nested_value
        elif isinstance(value, (int, float)):
            accumulated[key] = accumulated.get(key, 0) + value


def _partition_venues_by_geo(all_venues):
    """Partition residences into geo_unit Slices. Venues within a geo_unit are
    ordered by id for a stable per-Slice processing order (so RNG draws are
    fixed by the Slice, not by worker count). Builds the id->person index over
    every member so the parent can apply returned values."""
    geo_to_venues = defaultdict(list)
    id_to_person = {}
    for venue in all_venues:
        geo_unit = getattr(getattr(venue, 'geographical_unit', None),
                           'name', None) or '__no_geo__'
        geo_to_venues[geo_unit].append(venue)
        for member in venue.get_all_members():
            id_to_person[member.id] = member

    for venues in geo_to_venues.values():
        venues.sort(key=lambda venue: venue.id)
    slices = [GeoUnitSlice([geo_unit]) for geo_unit in sorted(geo_to_venues)]
    return slices, dict(geo_to_venues), id_to_person


def _assign_residence_slice(geo_slice, rng):
    """Worker: assign every residence in one geo_unit Slice and return compact
    arrays for the parent to apply. Seeds process-global np.random from the
    Slice's geo_unit identity (ADR-0002)."""
    context = _ACTIVE_ASSIGNMENT_CONTEXT
    assigner = context.assigner
    np.random.seed(int(rng.integers(0, 2 ** 32)))

    venues = []
    for geo_unit in geo_slice.unit_codes:
        venues.extend(context.geo_to_venues.get(geo_unit, []))
    residence_types = context.residence_venue_types
    households = [venue for venue in venues if venue.type in residence_types]
    other_residences = [venue for venue in venues if venue.type not in residence_types]

    # Distinct members across all residences in the Slice, for value collection.
    members = []
    seen_ids = set()
    for venue in venues:
        for member in venue.get_all_members():
            if member.id not in seen_ids:
                seen_ids.add(member.id)
                members.append(member)

    assigner.stats = assigner._fresh_stats()
    keys_before = [set(member.properties) for member in members]

    for household in households:
        assigner._assign_household(household)
    if other_residences:
        assigner._assign_other_residences(other_residences)

    return _collect_slice_result(assigner, members, keys_before)
