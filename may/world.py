"""
World module for MAY.

The World class is the main container for geography and population.
This module also contains setup functions for orchestrating world creation.
"""

import logging
from typing import Optional, Set
from may.residence.household_distributor import HouseholdDistributor, HouseholdError
from may.residence.allocation_strategy import execute_allocation_strategy
from may.utils import path_resolver as pr
from random import sample

logger = logging.getLogger("world")


class World:
    """
    The World object is the main container for a simulation.

    It holds references to the geography structure, population,
    and venues, along with any other world-level data needed for simulation.

    Attributes:
        geography (Geography): The geographical hierarchy
        population (PopulationManager): The population manager
        venues (VenueManager): The venue manager (includes all venues including households)
        household_distributor (HouseholdDistributor): The household distributor for allocation logic
    """

    def __init__(self, geography=None, population=None, venues=None, household_distributor=None):
        """
        Initialize a World object.

        Args:
            geography (Geography): Geography object containing geographical units
            population (PopulationManager): PopulationManager object containing people
            venues (VenueManager): VenueManager object containing all venues (including households)
            household_distributor (HouseholdDistributor): HouseholdDistributor for allocation logic (optional)
        """
        self.geography = geography
        self.population = population
        self.venues = venues
        self.household_distributor = household_distributor
        # WorldSerializer created by create_canonical_file(); reused by
        # export_to_hdf5() so the final export reopens the canonical file
        # in 'r+' instead of rewriting it. See docs/adr/0003.
        self._serializer = None

        # Register residence types from venue configuration with Person class
        if venues:
            from may.population.person import Person
            residence_types = venues.get_residence_types()
            Person.register_residence_types(residence_types)
            logger.info(f"Registered {len(residence_types)} residence types: {residence_types}")

    def get_households(self):
        """
        Get household-type residences (backwards compatible).

        Returns:
            List of Venue objects with type='household'
        """
        if self.venues:
            return self.venues.get_venues_by_type("household")
        return []

    def get_all_residences(self):
        """
        Get all residence venues (households, care homes, dorms, etc.).

        Returns:
            List of all residence Venue objects
        """
        if self.venues:
            return self.venues.get_all_residences()
        return []

    def get_residences_by_type(self, residence_type: str):
        """
        Get all residences of a specific type.

        Args:
            residence_type: Type of residence (e.g., 'care_home', 'prison', 'farm')

        Returns:
            List of Venue objects

        Example:
            >>> world.get_residences_by_type('care_home')
            [<Venue #0: care_home_0 (care_home) in E02000173>, ...]
        """
        if self.venues:
            return self.venues.get_venues_by_type(residence_type)
        return []

    def venues_by_type(self, venue_type: str):
        """
        Get all venues of a specific type.

        Args:
            venue_type: Type of venue (e.g., 'school', 'hospital', 'company')

        Returns:
            List of venues of the specified type
        """
        if self.venues:
            return self.venues.get_venues_by_type(venue_type)
        return []

    @property
    def people(self):
        """Convenience property to access all people in the population."""
        if self.population:
            return self.population.get_all_people()
        return []

    def __repr__(self):
        geo_str = f"{len(self.geography.get_all_units())} units" if self.geography else "no geography"
        pop_str = f"{len(self.population.get_all_people()):,} people" if self.population else "no population"

        if self.venues:
            total_venues = sum(len(d) for d in self.venues.venues_by_type_and_id.values())
            households = self.get_households()
            household_str = f"{len(households)} households"
            other_venues = total_venues - len(households)
            venue_str = f"{total_venues} venues ({household_str}, {other_venues} other)"
        else:
            venue_str = "no venues"

        return f"<World: {geo_str}, {pop_str}, {venue_str}>"

    def get_statistics(self):
        """
        Get comprehensive statistics about the world.

        Returns:
            dict: Dictionary containing geography, population, and venue statistics
        """
        stats = {}

        if self.geography:
            stats['geography'] = {
                'total_units': len(self.geography.get_all_units()),
                'units_by_level': {
                    level: len(self.geography.get_units_by_level(level))
                    for level in self.geography.levels
                }
            }

        if self.population:
            stats['population'] = self.population.get_statistics()

        if self.venues:
            stats['venues'] = {
                'total_venues': sum(len(d) for d in self.venues.venues_by_type_and_id.values()),
                'venue_types': len(self.venues.get_venue_types())
            }

        if self.household_distributor:
            households = self.get_households()
            total_allocated = len(self.household_distributor.allocated_people)
            total_people = sum(len(pool) for pool in self.household_distributor.person_pool_by_geo_unit.values())
            stats['households'] = {
                'total_households': len(households),
                'people_allocated': total_allocated,
                'people_unallocated': total_people - total_allocated,
                'allocation_rate': total_allocated / max(total_people, 1),
                'average_household_size': sum(h.size() for h in households) / max(len(households), 1)
            }

        return stats

    def assign_attributes(self, config_path: str, geo_units: Optional[Set[str]] = None,
                          workers: Optional[int] = None,
                          max_tasks_per_child: Optional[int] = None):
        """
        Assign attributes to all people in the world.

        This method uses the attribute assignment system to assign attributes
        (e.g., ethnicity) to all people based on YAML configuration.

        Args:
            config_path: Path to attribute assignment YAML config file
            geo_units: Optional set of geo unit codes to preload data for
            workers: Process-pool size, from the world config's
                `parallel.workers`. None keeps this stage's own default, which
                is the SLURM allocation and then the machine's CPU count.
            max_tasks_per_child: Recycling cadence, from the world config's
                `parallel.max_tasks_per_child`. None keeps today's behaviour
                of never retiring a worker.

        Returns:
            Dictionary with assignment statistics
        """
        from may.attribute_assignment import assign_attributes

        logger.info("")
        logger.info("="*60)
        logger.info(f"Assigning attributes...")
        logger.info("="*60)

        # Get geo units from geography if not provided
        # Include all hierarchy levels (SGU, MGU, LGU) for efficient filtering across all data sources
        geo_units_by_level = None
        if geo_units is None and self.geography:
            geo_units = set()
            # Same names, kept split by level. A source that declares which level
            # its values live at (an O-D matrix's `destination_level`) must be
            # matched against that level alone: names are not unique across
            # levels, so "Wales" as an XLGU would otherwise satisfy a check meant
            # for LGUs and send a cross-border commuter to a non-existent LAD.
            geo_units_by_level = {}
            for unit in self.geography.get_all_units_list():
                # Add the unit's name/code
                geo_units.add(unit.name)
                geo_units_by_level.setdefault(unit.level, set()).add(unit.name)
                # Also add parent names at all levels for O-D matrix filtering
                current = unit
                while current.parent:
                    geo_units.add(current.parent.name)
                    geo_units_by_level.setdefault(
                        current.parent.level, set()
                    ).add(current.parent.name)
                    current = current.parent

        # Run attribute assignment
        stats = assign_attributes(
            venue_manager=self.venues,
            config_path=config_path,
            geo_units=geo_units,
            geo_units_by_level=geo_units_by_level,
            population=self.population,
            workers=workers,
            max_tasks_per_child=max_tasks_per_child
        )

        return stats

    def create_canonical_file(self, output_file, config_file):
        """
        Create the canonical HDF5 file early (right after population
        generation) with core /population and /geography datasets, whose
        geo-sorted row order is already final. Later flush steps and the
        final export_to_hdf5() reopen this file in 'r+' and fill in the
        remaining datasets. See docs/adr/0003.

        Args:
            output_file: Path to output HDF5 file
            config_file: Path to serialization YAML config (required)
        """
        from may.serialization import WorldSerializer

        self._serializer = WorldSerializer(config_file=pr.resolve(config_file))
        self._serializer.export_core(self, output_file)

    def flush_properties(self, target, properties, venue_type=None, subsets=False):
        """
        Flush settled properties, or venue subset membership, to the
        canonical HDF5 file mid-timeline and free their in-memory values
        (a `flush_properties` timeline step).

        After a successful property flush the keys are tombstoned: `in`
        still returns True but reading or writing the value raises
        FlushedPropertyError (see may/flushing.py). After a successful
        subsets flush, each subset's members are staged to disk and its
        `.members`/`.member_metadata` become gutted sentinels raising
        FlushedPropertyError. The final export skips flushed columns and
        streams staged subset membership back in from disk.

        A requested property whose every live value is `None` is deferred:
        nothing is written, it is not tombstoned, and it may be requested
        again by a later flush_properties call or picked up at final
        export once it has real values.

        When both `properties` and `subsets` are given, requested property
        names are validated up front (no I/O), then subsets are flushed
        before properties. This makes a call that fails partway safe to
        retry identically: if subsets fails, properties were never touched;
        if properties fails after subsets already succeeded, the retry's
        subsets flush is a no-op (identical keys already recorded) and only
        properties runs for real.

        Args:
            target: 'population', 'geo_units' or 'venues'
            properties: list of property names to flush; may be empty iff
                `subsets` is True
            venue_type: required for target 'venues' (the cohort to
                flush); must be omitted for other targets
            subsets: flush this venue_type's subset membership to staging;
                only valid with target 'venues'
        """
        valid_targets = {"population", "geo_units", "venues"}
        if target not in valid_targets:
            raise ValueError(
                f"flush_properties target {target!r} invalid; expected one "
                f"of {sorted(valid_targets)}."
            )
        if subsets and target != "venues":
            raise ValueError(
                "flush_properties `subsets: true` is only valid with "
                "target 'venues'."
            )
        if not properties and not subsets:
            raise ValueError(
                "flush_properties requires a non-empty `properties:` list "
                "(unless `subsets: true` is set)."
            )
        if properties and not isinstance(properties, list):
            raise ValueError(
                "flush_properties requires a non-empty `properties:` list."
            )
        if target == "venues" and not venue_type:
            raise ValueError(
                "flush_properties target 'venues' requires `venue_type:`."
            )
        if target != "venues" and venue_type is not None:
            raise ValueError(
                f"flush_properties target {target!r} does not take `venue_type:`."
            )
        if self._serializer is None:
            raise RuntimeError(
                "flush_properties requires the canonical HDF5 file: "
                "serialization must be enabled and create_canonical_file() "
                "called before the timeline."
            )

        from may.flushing import tombstone_objects
        from may.serialization.flush_specs import FLUSH_SPECS

        if properties:
            spec = FLUSH_SPECS[target]
            spec.tracker(self, venue_type).validate(
                properties, spec.configured(self, venue_type)
            )

        if subsets:
            self._serializer.flush_venue_subsets(self, venue_type)

        if properties:
            spec = FLUSH_SPECS[target]
            written_properties = spec.serializer_call(self, venue_type, properties)
            tombstone_objects(
                spec.cohort(self, venue_type), written_properties,
                spec.registry(self, venue_type),
            )

    def flush_subsets_batch(self, venue_type, subset_keys):
        """Stage one batch of `venue_type` subsets and gut them.

        The batch form of `flush_properties(subsets=True)`, for callers that
        flush a venue_type incrementally rather than all at once — residence
        allocation stages each geo unit's households as that unit finishes.
        Batches must be disjoint; the staged keys accumulate and their union
        is verified at final export.
        """
        if self._serializer is None:
            raise RuntimeError(
                "flush_subsets_batch requires the canonical HDF5 file: "
                "serialization must be enabled and create_canonical_file() "
                "called before the timeline."
            )
        self._serializer.flush_venue_subsets(self, venue_type,
                                             subset_keys=subset_keys)

    def export_to_hdf5(self, output_file, config_file=None):
        """
        Export world state to HDF5 file for C++ simulation.

        This method serializes the complete world state (geography, population,
        venues, and relationships) to an HDF5 file that can be efficiently loaded
        by the C++ simulation engine.

        The serialization configuration (config_file) determines which extra
        properties are included in the export. If create_canonical_file() was
        already called on this world, its config is reused: config_file may be
        omitted, or if given must resolve to that same config, otherwise a
        ValueError is raised. Otherwise config_file is required: there is no
        scenario-generic default schema.

        Args:
            output_file: Path to output HDF5 file
            config_file: Path to serialization YAML config. Required unless
                create_canonical_file() was already called on this world, in
                which case it must match that config if given at all.

        Returns:
            dict: Export statistics (num_people, num_venues, etc.)

        Example:
            >>> world.export_to_hdf5("world_state.h5", "configs/2021/serialization_config.yaml")
        """
        from may.serialization import WorldSerializer

        logger.info("")
        logger.info("=" * 60)
        logger.info("EXPORTING WORLD TO HDF5")
        logger.info("=" * 60)

        if self._serializer is not None:
            if config_file is not None:
                resolved_config_file = pr.resolve(config_file)
                if resolved_config_file != self._serializer.config.config_file:
                    raise ValueError(
                        f"export_to_hdf5 config_file {resolved_config_file!r} does "
                        f"not match config {self._serializer.config.config_file!r} "
                        "used by create_canonical_file()"
                    )
            serializer = self._serializer
        else:
            serializer = WorldSerializer(config_file=pr.resolve(config_file))
        stats = serializer.export(self, output_file)

        logger.info("")
        logger.info(f"Successfully exported world to: {output_file}")
        logger.info("=" * 60)

        return stats


def _household_membership_is_staged(households):
    """True if household subsets have been gutted by a per-unit flush, so
    their members live in staging rather than memory."""
    from may.flushing import GuttedMembers

    for household in households:
        for subset in household.subsets.values():
            return isinstance(subset.members, GuttedMembers)
    return False


def resolve_configured_workers(config):
    """Pool size from the world config's `parallel:` block, or None.

    One dial for both parallelised stages — residence allocation and attribute
    assignment — because both fork from the same process on the same node, and
    an operator sizing a job from its SLURM allocation is answering one
    question, not two.

    None means "leave each stage on its own default", which is deliberately not
    the same for both: attribute assignment takes the SLURM allocation or the
    machine, residence allocation takes one worker until its per-worker
    footprint has been measured at scale. A run that has asked for nothing
    should not have either moved under it.
    """
    workers = (config or {}).get("parallel", {}).get("workers")
    if workers is None:
        return None
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise HouseholdError(
            f"parallel.workers must be a positive whole number of worker "
            f"processes, not {workers!r}; omit the key to leave each stage on "
            f"its own default"
        )
    return workers


def resolve_configured_max_tasks_per_child(config):
    """Recycling cadence from the world config's `parallel:` block, or None.

    Sibling to `resolve_configured_workers`, same one-dial-for-both-stages
    reasoning: a forked worker is reused for every Slice it draws for the
    life of the pool, and nothing resets what it accumulated in between
    (`CONTEXT.md`'s "Worker world copy"). Neither stage's parent-side memory
    defence — residence's `memory_budget_mb` flush, attribute assignment's
    plain streamed apply — sees or bounds that worker-side growth.

    None means "never retire a worker", which is today's behaviour for both
    stages. A run that has asked for nothing should not have that moved
    under it.
    """
    max_tasks_per_child = (config or {}).get("parallel", {}).get("max_tasks_per_child")
    if max_tasks_per_child is None:
        return None
    if (isinstance(max_tasks_per_child, bool)
            or not isinstance(max_tasks_per_child, int)
            or max_tasks_per_child < 1):
        raise HouseholdError(
            f"parallel.max_tasks_per_child must be a positive whole number "
            f"of Slices, not {max_tasks_per_child!r}; omit the key to never "
            f"retire a worker"
        )
    return max_tasks_per_child


def setup_households(geo, population, venues, config, strategy_file=None,
                     world=None, workers=None, max_tasks_per_child=None):
    """
    Setup and distribute households based on configuration.

    This orchestration function:
    - Creates a HouseholdDistributor
    - Loads household data
    - Executes allocation strategy
    - Exports venue allocations

    Args:
        geo: Geography object
        population: PopulationManager object
        venues: VenueManager object
        config: Configuration dictionary
        strategy_file: Path to the allocation-strategy YAML to execute. This is
            the `config:` of the timeline's `residence_allocation` step, the
            single source of truth for which strategy runs and where it runs in
            the timeline. When ``None`` (e.g. a direct programmatic caller),
            falls back to ``config["households"]["strategy_file"]``.
        world: The World being built, when one exists. Supplies the subset
            flusher a `memory_budget_mb` strategy needs; without it
            such a strategy is a config error.
        workers: Process-pool size for the geo-unit loop, from the world
            config's `parallel.workers`. None keeps this stage's own default
            of one worker — it forks a whole world per worker, and its
            per-worker footprint has not been measured at scale.
        max_tasks_per_child: Recycling cadence for the geo-unit loop's pool,
            from the world config's `parallel.max_tasks_per_child`. None
            keeps today's behaviour of never retiring a worker.

    Returns:
        HouseholdDistributor object with allocated households
    """
    logger.info("")
    logger.info("Distributing households...")
    household_config = config.get("households", {})

    household_distributor = HouseholdDistributor(
        geography=geo,
        population=population,
        venue_manager=venues,
        data_dir=pr.resolve(household_config.get("data_dir", "data/households")),
        config_file=pr.resolve(household_config.get("config_file", "households_config.yaml")),
        rules_file=pr.resolve(household_config.get("rules_file")) if household_config.get("rules_file") else None,
    )

    # Load household data. data_file may be a single file or a list of files
    # to stack; column_policy: union_zero_fill lets sources with different
    # composition vocabularies combine (absent pattern = zero households).
    household_data_file = household_config.get("data_file", "households.csv")
    household_distributor.load_household_data(
        household_data_file,
        column_policy=household_config.get("column_policy", "strict"),
    )

    # Distribute households and venues based on configuration mode.
    # The strategy file comes from the residence_allocation timeline step;
    # fall back to the households block only for direct programmatic callers.
    if strategy_file is None and household_config.get("strategy_file"):
        strategy_file = pr.resolve(household_config.get("strategy_file"))

    debug_outputs_enabled = config.get("debug_outputs", {}).get("enabled", False)

    # Only offered when serialization is live; the strategy raises if it set
    # a pending-flush bound and there is nothing to stage into.
    subset_flusher = (world.flush_subsets_batch
                      if world is not None and world._serializer is not None
                      else None)

    if strategy_file:
        # Mode 1: Unified strategy (households + venues in order)
        logger.info(f"Using unified allocation strategy from {strategy_file}")
        execute_allocation_strategy(
            population,
            venues,
            household_distributor,
            strategy_file,
            export_debug_csv=debug_outputs_enabled,
            subset_flusher=subset_flusher,
            workers=workers if workers is not None else 1,
            max_tasks_per_child=max_tasks_per_child,
        )

    # Show where households are located and examples
    logger.info("")
    logger.info("=" * 60)
    logger.info("HOUSEHOLD STORAGE LOCATIONS")
    logger.info("=" * 60)

    # Get all households from VenueManager
    all_households = venues.get_venues_by_type("household")
    logger.info(f"Total households created: {len(all_households):,}")
    logger.info("")

    logger.info("Households are stored in VenueManager:")
    logger.info("  1. venues.get_venues_by_type('household')  -> List of all household Venues")
    logger.info("  2. venues.get_venue_by_type_and_id('household', id)  -> Specific household by ID")
    logger.info("")

    # The illustrative logging below walks household members, which a run
    # with a pending-flush bound has already staged and freed. Skip it rather
    # than read membership back off disk purely to print five examples.
    membership_staged = _household_membership_is_staged(all_households)
    if membership_staged:
        logger.info("Household membership was flushed to staging during "
                    "allocation; skipping example households and residents.")

    # Show a few example households
    if all_households and not membership_staged:
        logger.info("Example Households (5 random):")
        n = len(all_households)
        sampled_indices = set(sample(range(n), min(5, n)))
        for i, household in enumerate(all_households):
            if i not in sampled_indices:
                continue
            age_categories = household.properties.get('_age_categories', [])
            composition = household.get_composition(age_categories)
            members = household.get_all_members()

            logger.info(f"")
            logger.info(f"  Household ID: {household.id} (type-scoped ID)")
            logger.info(f"  Venue ID: {id(household)} (Python object ID)")
            logger.info(f"  Name: {household.name}")
            logger.info(f"  Type: {household.type}")
            logger.info(f"  Location: {household.geographical_unit.name}")
            logger.info(f"  Size: {household.size()} people")
            logger.info(f"  Composition: {composition}")
            if members:
                logger.info(f"  Members: {', '.join([f'Person_{p.id}({p.age}{'m' if p.sex=='male' else 'f'})' for p in members])}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("PEOPLE IN HOUSEHOLDS - How to find where someone lives")
    logger.info("=" * 60)

    # Show how to access people's households
    if household_distributor.allocated_people and not membership_staged:
        example_person_ids = sample(list(household_distributor.allocated_people), 5)

        for person_id in example_person_ids:
            person = population.get_person(person_id)
            # UNIFIED STRUCTURE: activity_map['residence']['household'] = [subsets]
            if person and "residence" in person.activity_map and "household" in person.activity_map["residence"]:
                household_subsets = person.activity_map["residence"]["household"]
                if household_subsets:
                    household_venue = household_subsets[0].venue
                    age_categories = household_venue.properties.get('_age_categories', [])

                    logger.info(f"")
                    logger.info(f"  Person {person.id} (age={person.age}, sex={person.sex})")
                    logger.info(f"  Activity map: {person.activity_map}")
                    logger.info(f"  Lives in: {household_venue.name} (ID={household_venue.id})")
                    logger.info(f"  Location: {household_venue.geographical_unit.name}")
                    logger.info(f"  Household size: {household_venue.size()}")
                    logger.info(f"  Household composition: {household_venue.get_composition(age_categories)}")

    logger.info("")
    logger.info("=" * 60)

    return household_distributor
