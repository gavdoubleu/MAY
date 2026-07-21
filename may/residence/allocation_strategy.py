"""
Unified allocation strategy executor.

Executes both household and venue allocations in a single YAML-defined sequence.
"""

import os
import hashlib
import logging
import yaml
import numpy as np
from itertools import islice
from typing import Callable, Dict, List, Optional, Set, Tuple
from .venue_allocator import _allocate_to_venue_type
from .household_distributor import HouseholdError
from may.utils import path_resolver as pr

logger = logging.getLogger("allocation_strategy")

# The venue type per-unit flushing stages: households are what allocation
# creates, and what dominates peak memory
HOUSEHOLD_VENUE_TYPE = 'household'


def _household_venue_count(venues) -> int:
    return len(venues.venues_by_type_and_id.get(HOUSEHOLD_VENUE_TYPE, {}))


def _new_household_subset_keys(venues, count_before_unit, scope_sgu_codes,
                               unit_code) -> List[Tuple[int, int]]:
    """Subset keys of the households created during the geo unit that just
    finished.

    Households are appended to the venue manager's per-type dict in creation
    order, so the unit's own are its tail — walked backwards so the cost is
    in the new households, not the whole world's. Each one's geo unit is
    checked against the unit's scope: if anything ever removes a household
    mid-run the tail arithmetic would silently pick the wrong venues, and
    staging the wrong membership is not a failure worth discovering later.
    """
    households = venues.venues_by_type_and_id.get(HOUSEHOLD_VENUE_TYPE, {})
    new_household_count = len(households) - count_before_unit
    subset_keys = []
    for venue in islice(reversed(households.values()), new_household_count):
        if venue.geographical_unit.name not in scope_sgu_codes:
            raise HouseholdError(
                f"Household venue {venue.id} sits in "
                f"'{venue.geographical_unit.name}', outside geo unit "
                f"'{unit_code}', but was counted among the {new_household_count} "
                f"households it created — the venue manager's household "
                f"ordering cannot be trusted to stage per-unit batches."
            )
        subset_keys.extend((venue.id, subset.subset_index)
                           for subset in venue.subsets.values())
    return subset_keys


def execute_allocation_strategy(population,
                                venues,
                                household_distributor,
                                strategy_file: str = "data/households/allocation_strategy.yaml",
                                export_debug_csv: bool = False,
                                random_seed: Optional[int] = None,
                                subset_flusher: Optional[Callable] = None):
    """
    Execute a unified allocation strategy from YAML configuration.

    This function orchestrates BOTH household and venue allocations. The
    outer loop runs over geographical units at the strategy's `loop_level`
    (default: the largest configured level); the full step sequence runs per
    unit, restricted to that unit's smallest-level units. Each unit draws
    from its own random stream, so results are independent of the unit
    iteration order.

    Args:
        population: PopulationManager
        venues: VenueManager
        household_distributor: HouseholdDistributor
        strategy_file (str, optional): Path to YAML strategy file (relative or absolute). Default is "data/households/allocation_strategy.yaml".
        random_seed: Global seed for the per-unit random streams (each unit's
            stream is spawned from SeedSequence((random_seed, digest(unit)))).
            When None, seed 0 is used — matching the historical default.
        subset_flusher: Callable (venue_type, subset_keys) staging a batch of
            subsets and gutting them. Required when the strategy sets
            `flush_per_unit: true`; called once per geo unit with the
            household subsets that unit created.

    Returns:
        dict: Complete statistics for all steps, aggregated across units

    """
    logger.info("=" * 60)
    logger.info("Executing Unified Allocation Strategy")
    logger.info("=" * 60)

    # Resolve template variables, then fall back to data/ prefix for bare relative paths
    strategy_file = pr.resolve(strategy_file)
    if not os.path.isabs(strategy_file):
        if not os.path.exists(strategy_file):
            strategy_file = f"data/{strategy_file}"

    # Load strategy configuration
    logger.info(f"Loading allocation strategy from {strategy_file}")
    with open(strategy_file, 'r') as f:
        strategy = yaml.safe_load(f)

    # Check if enabled
    if not strategy.get('enabled', True):
        logger.info("Unified strategy is disabled, skipping")
        return {}

    # Get steps
    steps = strategy.get('steps', [])
    if not steps:
        logger.warning("No allocation steps defined in strategy")
        return {}

    geography = household_distributor.geography

    # Outer-loop geography level: default is the largest configured level
    loop_level = strategy.get('loop_level') or geography.levels[-1]
    if loop_level not in geography.levels:
        raise HouseholdError(
            f"Unknown loop_level '{loop_level}' in {strategy_file}; "
            f"configured geography levels: {geography.levels}"
        )

    flush_per_unit = strategy.get('flush_per_unit', False)
    if flush_per_unit and subset_flusher is None:
        raise HouseholdError(
            f"flush_per_unit: true in {strategy_file} requires serialization "
            f"to be enabled: staged subsets are written beside the canonical "
            f"HDF5 file, which does not exist for this run"
        )

    _validate_step_patterns(steps, household_distributor.household_pattern_vocabulary)
    _validate_venue_step_levels(steps, venues, geography, loop_level)

    # Filter to known step types once, warning per step rather than per unit
    known_step_types = ['household', 'venue', 'household_excess',
                        'household_overflow', 'household_promotion']
    numbered_steps: List[Tuple[int, str, str, Dict]] = []
    step_number = 1
    for step_config in steps:
        step_type = step_config.get('type')
        step_name = step_config.get('name', f'Step {step_number}')
        if step_type not in known_step_types:
            logger.warning(f"Unknown step type '{step_type}' for step '{step_name}', skipping")
            continue
        numbered_steps.append((step_number, step_name, step_type, step_config))
        step_number += 1

    smallest_level = geography.levels[0]
    loop_units = geography.get_units_by_level(loop_level)
    global_seed = random_seed if random_seed is not None else 0

    logger.info(f"Found {len(numbered_steps)} allocation steps")
    logger.info(f"Looping over {len(loop_units)} {loop_level} unit(s)")
    logger.info("")

    all_stats: Dict[str, Dict] = {}
    total_units = len(loop_units)
    progress_interval = max(1, total_units // 10)  # Update every 10%

    for unit_index, (unit_code, unit) in enumerate(loop_units.items(), 1):
        # Per-unit random stream: derived from the global seed and a stable
        # digest of the unit code, so results are independent of unit order
        household_distributor.set_random_generator(np.random.default_rng(
            np.random.SeedSequence((global_seed, _stable_unit_digest(unit_code)))))

        if unit.level == smallest_level:
            scope_sgu_codes = {unit_code}
        else:
            scope_sgu_codes = {descendant.name for descendant
                               in unit.get_descendants(level=smallest_level)}
        # The unit's whole subtree (any level) — venues may sit between the
        # loop level and the smallest level
        scope_geo_unit_names = {unit_code} | {descendant.name for descendant
                                              in unit.get_descendants()}
        household_distributor.set_geo_unit_scope(scope_sgu_codes)

        household_count_before_unit = _household_venue_count(venues)

        logger.info("=" * 60)
        logger.info(f"Geo unit {unit_index}/{total_units}: {unit_code} "
                    f"({len(scope_sgu_codes)} {smallest_level} unit(s))")
        logger.info("=" * 60)

        for step_number, step_name, step_type, step_config in numbered_steps:
            logger.info("-" * 60)
            logger.info(f"[{unit_code}] Step {step_number}: {step_name} ({step_type})")
            logger.info("-" * 60)

            description = step_config.get('description')
            if description:
                logger.info(f"Description: {description}")
                logger.info("")

            # Execute based on type
            if step_type == 'household':
                stats = _execute_household_step(step_config, household_distributor)
            elif step_type == 'venue':
                stats = _execute_venue_step(step_config, population, venues,
                                            household_distributor,
                                            scope_geo_unit_names, scope_sgu_codes)
            elif step_type == 'household_excess':
                stats = _execute_household_excess_step(step_config, household_distributor)
            elif step_type == 'household_overflow':
                stats = _execute_household_overflow_step(step_config, household_distributor)
            elif step_type == 'household_promotion':
                stats = _execute_household_promotion_step(step_config, household_distributor)

            accumulated = all_stats.setdefault(
                step_name, {'type': step_type, 'step_number': step_number})
            _merge_step_stats(accumulated, stats)
            logger.info("")

        if flush_per_unit:
            subset_keys = _new_household_subset_keys(
                venues, household_count_before_unit, scope_sgu_codes, unit_code)
            if subset_keys:
                subset_flusher(HOUSEHOLD_VENUE_TYPE, subset_keys)
                logger.info(f"[{unit_code}] Flushed {len(subset_keys)} household "
                            f"subsets to staging")

        if unit_index % progress_interval == 0 or unit_index == total_units:
            percent_complete = (unit_index / total_units) * 100
            logger.info(f"Geo-unit progress: {unit_index}/{total_units} "
                        f"({percent_complete:.1f}%) - "
                        f"{len(household_distributor.allocated_people):,} people allocated")

    # Clear the scope so any later use of the distributor sees the whole world
    household_distributor.set_geo_unit_scope(None)

    # Print overall summary
    logger.info("=" * 60)
    logger.info("UNIFIED ALLOCATION STRATEGY SUMMARY")
    logger.info("=" * 60)
    logger.info("")

    total_household_alloc = 0
    total_venue_alloc = 0
    total_excess_alloc = 0
    total_overflow_alloc = 0

    for step_name, stats in all_stats.items():
        logger.info(f"{stats['step_number']}. {step_name} ({stats['type']}):")

        if stats['type'] == 'household':
            households_created = stats.get('households_created', 0)
            people_allocated = stats.get('people_allocated_this_round', 0)
            logger.info(f"   Households: {households_created:,}")
            logger.info(f"   People: {people_allocated:,}")
            total_household_alloc += people_allocated

        elif stats['type'] == 'venue':
            allocated = stats.get('allocated', 0)
            venues_count = stats.get('venues', 0)
            logger.info(f"   Venues: {venues_count}")
            logger.info(f"   People: {allocated:,}")
            total_venue_alloc += allocated

        elif stats['type'] == 'household_excess':
            people_added = stats.get('people_added', 0)
            households_modified = stats.get('households_modified', 0)
            logger.info(f"   Households modified: {households_modified:,}")
            logger.info(f"   People added: {people_added:,}")
            total_excess_alloc += people_added

        elif stats['type'] == 'household_overflow':
            people_added = stats.get('people_added', 0)
            households_modified = stats.get('households_modified', 0)
            logger.info(f"   Households modified: {households_modified:,}")
            logger.info(f"   People added (overflow): {people_added:,}")
            total_overflow_alloc += people_added

        elif stats['type'] == 'household_promotion':
            people_added = stats.get('people_added', 0)
            households_promoted = stats.get('households_promoted', 0)
            logger.info(f"   Households promoted: {households_promoted:,}")
            logger.info(f"   People added (promotion): {people_added:,}")
            total_overflow_alloc += people_added  # Count with overflow

        logger.info("")

    logger.info("Overall Totals:")
    # Get household count from VenueManager
    all_households = household_distributor.venue_manager.get_venues_by_type("household")
    logger.info(f"  Total households: {len(all_households):,}")
    logger.info(f"  People in households (initial): {total_household_alloc:,}")
    logger.info(f"  People added to households (excess): {total_excess_alloc:,}")
    logger.info(f"  People added to households (overflow): {total_overflow_alloc:,}")
    logger.info(f"  People in venues: {total_venue_alloc:,}")

    logger.info(f"  Total allocated: {len(household_distributor.allocated_people):,}")
    logger.info(f"  Remaining unallocated: {household_distributor.get_available_people_count():,}")

    total_pop = len(population.get_all_people())
    alloc_pct = (len(household_distributor.allocated_people) / total_pop * 100) if total_pop > 0 else 0
    logger.info(f"  Allocation rate: {alloc_pct:.1f}%")
    logger.info("=" * 60)

    # Optionally export unallocated people (skipped for large worlds —
    # builds a DataFrame across every unplaced person).
    if export_debug_csv:
        household_distributor.export_unallocated_people_to_csv()

    return all_stats


def _stable_unit_digest(unit_code: str) -> int:
    """Platform-independent 64-bit digest of a geo-unit code for seeding.

    Python's hash() is randomised per process, so it cannot seed reproducible
    per-unit streams.
    """
    return int.from_bytes(
        hashlib.blake2b(unit_code.encode('utf-8'), digest_size=8).digest(), 'big')


# Stats keys that are world-level snapshots rather than per-unit counts:
# merging keeps the latest value instead of summing
SNAPSHOT_STAT_KEYS = {'round_name', 'round_number', 'total_households',
                      'total_people_allocated', 'total_people_remaining',
                      'error'}


def _merge_step_stats(accumulated: Dict, unit_stats: Dict) -> None:
    """Fold one unit's step stats into the running totals for that step.

    Counts add across units, snapshot keys keep the latest value, nested count
    dicts (e.g. allocation_by_attribute) add per key, and capacity_pct is
    recomputed from the merged counts.
    """
    for key, value in unit_stats.items():
        if key == 'capacity_pct':
            continue  # derived — recomputed below
        if isinstance(value, dict):
            nested = accumulated.setdefault(key, {})
            for nested_key, nested_value in value.items():
                nested[nested_key] = nested.get(nested_key, 0) + nested_value
        elif (key in SNAPSHOT_STAT_KEYS or isinstance(value, (str, bool))
              or value is None):
            accumulated[key] = value
        elif isinstance(value, (int, float)):
            accumulated[key] = accumulated.get(key, 0) + value
        else:
            accumulated[key] = value
    if 'total_capacity' in accumulated:
        total_capacity = accumulated['total_capacity']
        accumulated['capacity_pct'] = (
            accumulated.get('capacity_used', 0) / total_capacity * 100
            if total_capacity else 0)


def _validate_venue_step_levels(steps: List[Dict], venues, geography,
                                loop_level: str) -> None:
    """Fail loud when a venue step targets venues above the loop level.

    Such venues span multiple loop units, so per-unit allocation cannot fill
    them coherently. Revisit if a real config needs coarse venues.
    """
    loop_level_index = geography.levels.index(loop_level)
    offending: Dict[str, Set[str]] = {}
    for step in steps:
        if step.get('type') != 'venue':
            continue
        venue_type = step.get('venue_type')
        for venue in venues.get_venues_by_type(venue_type):
            venue_level = venue.geographical_unit.level
            if geography.levels.index(venue_level) > loop_level_index:
                offending.setdefault(venue_type, set()).add(venue_level)
    if offending:
        raise HouseholdError(
            f"Venue step(s) target venues above loop_level '{loop_level}': "
            f"{ {vt: sorted(levels) for vt, levels in offending.items()} }. "
            f"Raise loop_level or move these venues to a finer level."
        )


def _validate_step_patterns(steps: List[Dict], vocabulary: set) -> None:
    """Fail loud on a `household` build-step pattern absent from households.csv.

    A build step's matcher iterates the CSV columns, so a pattern that isn't a
    column never appears — it builds nothing with no log at all (the silent
    exact-string foot-gun: typo / stray space / wrong operator). Only build steps
    are checked: `household_excess`/`household_overflow` `target_patterns` are a
    catch-all superset matched against existing households' original_pattern and
    already warn ("matched no households") on a miss, and `household_promotion`
    source_patterns match a runtime allocation_pattern, not the CSV.
    """
    if not vocabulary:
        return  # no data loaded to validate against
    unknown: Dict[str, List[str]] = {}
    for step in steps:
        if step.get('type') != 'household':
            continue
        names = [p.get('pattern') if isinstance(p, dict) else p
                 for p in (step.get('patterns') or [])]
        for name in names:
            if name and name not in vocabulary:
                unknown.setdefault(step.get('name', 'household'), []).append(name)
    if unknown:
        raise HouseholdError(
            f"household build step(s) reference composition patterns absent from "
            f"households.csv: {unknown}. Known patterns: {sorted(vocabulary)}"
        )


def _execute_household_step(step_config: Dict, household_distributor) -> Dict:
    """
    Execute a household allocation step.

    Args:
        step_config: Configuration dict for this step
        household_distributor: HouseholdDistributor

    Returns:
        dict: Statistics for this step
    """
    patterns = step_config.get('patterns')
    max_households = step_config.get('max_households')
    refresh_pools = step_config.get('refresh_pools', False)
    enable_demotion = step_config.get('enable_demotion')
    max_household_size = step_config.get('max_household_size')
    allocate_flexible = step_config.get('allocate_flexible', False)
    round_name = step_config.get('name', 'Household Round')
    rule_name = step_config.get('rule')  # Optional: explicit rule to apply
    demotion_rules = step_config.get('demotion_rules', {})  # Optional: pattern -> rule mapping for demotions

    if rule_name:
        logger.info(f"  Using explicit relationship rule: '{rule_name}'")

    if demotion_rules:
        logger.info(f"  Demotion rules configured: {len(demotion_rules)} pattern(s)")

    # Process patterns to extract assumptions
    # Patterns can be either:
    #   - Simple strings: "0 >=0 0 0"
    #   - Dicts with pattern and assumption: {pattern: "0 >=0 0 0", assumption: "0 2 0 0"}
    pattern_list = None
    pattern_assumptions = {}

    if patterns is not None:
        pattern_list = []
        for p in patterns:
            if isinstance(p, dict):
                # Format with assumption
                pattern_str = p.get('pattern')
                assumption_str = p.get('assumption')

                if pattern_str:
                    pattern_list.append(pattern_str)
                    if assumption_str:
                        pattern_assumptions[pattern_str] = assumption_str
                        logger.info(f"  Pattern '{pattern_str}' has assumption: '{assumption_str}'")
            else:
                # Simple format
                pattern_list.append(p)

    # Temporarily override demotion if specified
    original_demotion = None
    if enable_demotion is not None:
        original_demotion = household_distributor.config['demotion']['enabled']
        household_distributor.config['demotion']['enabled'] = enable_demotion

    try:
        stats = household_distributor.round_distributor.distribute_households_round(
            pattern_filter=pattern_list,
            pattern_assumptions=pattern_assumptions,
            max_households=max_households,
            max_household_size=max_household_size,
            allocate_flexible=allocate_flexible,
            refresh_pools=refresh_pools,
            round_name=round_name,
            rule_name=rule_name,
            demotion_rules=demotion_rules
        )
        return stats
    finally:
        # Restore original demotion setting
        if original_demotion is not None:
            household_distributor.config['demotion']['enabled'] = original_demotion


def _execute_household_excess_step(step_config: Dict, household_distributor) -> Dict:
    """
    Execute a household excess allocation step.

    This step adds people to existing households created in previous steps.

    Args:
        step_config: Configuration dict for this step
        household_distributor: HouseholdDistributor

    Returns:
        dict: Statistics for this step
    """
    target_patterns = step_config.get('target_patterns', [])
    add_category = step_config.get('add_category')
    constraints = step_config.get('constraints')
    max_per_household = step_config.get('max_per_household')
    add_distribution = step_config.get('add_distribution')
    refresh_pools = step_config.get('refresh_pools', False)
    round_name = step_config.get('name', 'Household Excess Round')
    rule_name = step_config.get('rule')  # Optional: relationship rule to apply

    if rule_name:
        logger.info(f"  Using explicit relationship rule: '{rule_name}'")

    if not add_category:
        logger.error("No 'add_category' specified for household_excess step")
        return {
            'people_added': 0,
            'households_modified': 0,
            'error': "Missing 'add_category' parameter"
        }

    stats = household_distributor.allocate_excess_to_households(
        target_patterns=target_patterns,
        add_category=add_category,
        constraints=constraints,
        max_per_household=max_per_household,
        add_distribution=add_distribution,
        refresh_pools=refresh_pools,
        round_name=round_name,
        rule_name=rule_name
    )

    return stats


def _execute_household_overflow_step(step_config: Dict, household_distributor) -> Dict:
    """
    Execute a household overflow allocation step.

    This step adds ALL remaining people from a category to existing households,
    IGNORING max household size constraints. People are distributed balancedly
    across eligible households with optional pattern biasing.

    Args:
        step_config: Configuration dict for this step
        household_distributor: HouseholdDistributor

    Returns:
        dict: Statistics for this step
    """
    target_patterns = step_config.get('target_patterns', [])
    add_category = step_config.get('add_category')
    pattern_bias = step_config.get('pattern_bias', {})  # e.g., {"0 >=0 0 0": 2.0}
    refresh_pools = step_config.get('refresh_pools', False)
    round_name = step_config.get('name', 'Household Overflow Round')

    if not add_category:
        logger.error("No 'add_category' specified for household_overflow step")
        return {
            'people_added': 0,
            'households_modified': 0,
            'error': "Missing 'add_category' parameter"
        }

    stats = household_distributor.allocate_overflow_to_households(
        target_patterns=target_patterns,
        add_category=add_category,
        pattern_bias=pattern_bias,
        refresh_pools=refresh_pools,
        round_name=round_name
    )

    return stats


def _execute_household_promotion_step(step_config: Dict, household_distributor) -> Dict:
    """
    Execute a household promotion allocation step.

    This step promotes existing households according to specific rules,
    allowing controlled acceptance of remaining people.

    Args:
        step_config: Configuration dict for this step
        household_distributor: HouseholdDistributor

    Returns:
        dict: Statistics for this step
    """
    promotion_rules = step_config.get('promotion_rules', [])
    target_categories = step_config.get('target_categories', [])  # Fallback to simple mode
    refresh_pools = step_config.get('refresh_pools', False)
    round_name = step_config.get('name', 'Household Promotion Round')

    if not promotion_rules and not target_categories:
        logger.error("No 'promotion_rules' or 'target_categories' specified for household_promotion step")
        return {
            'people_added': 0,
            'households_promoted': 0,
            'error': "Missing 'promotion_rules' or 'target_categories' parameter"
        }

    if promotion_rules:
        # Rule-based promotion (controlled)
        stats = household_distributor.promote_with_rules(
            promotion_rules=promotion_rules,
            refresh_pools=refresh_pools,
            round_name=round_name
        )
    else:
        # Simple promotion (all categories)
        stats = household_distributor.promote_and_allocate(
            target_categories=target_categories,
            refresh_pools=refresh_pools,
            round_name=round_name
        )

    return stats


def _execute_venue_step(step_config: Dict, population, venues, household_distributor,
                        scope_geo_unit_names: Optional[Set[str]] = None,
                        scope_sgu_codes: Optional[Set[str]] = None) -> Dict:
    """
    Execute a venue allocation step.

    Args:
        step_config: Configuration dict for this step
        population: PopulationManager
        venues: VenueManager
        household_distributor: HouseholdDistributor
        scope_geo_unit_names: Geo-unit names in the current loop unit's
            subtree (restricts venues); None = whole world
        scope_sgu_codes: Smallest-level unit codes in the current loop unit
            (restricts people); None = whole world

    Returns:
        dict: Statistics for this step
    """
    venue_type = step_config.get('venue_type')

    # max_allocations was a global, order-dependent stop condition removed
    # outright — per-venue capacities are the mechanism. Warn on stale configs.
    if 'max_allocations' in step_config:
        logger.warning(
            f"  'max_allocations' is no longer supported and is ignored "
            f"(venue step '{step_config.get('name')}') — remove it from the "
            f"strategy config; per-venue capacities cap allocation"
        )

    # Create allocation config for venue allocator
    allocation_config = {
        'venue_type': venue_type,
        'description': step_config.get('description', ''),
        'capacity_property': step_config.get('capacity_property', 'capacity'),
        'eligibility': step_config.get('eligibility', {}),
        'strategy': step_config.get('strategy', 'random'),
        # Capacity rules owned by this allocation step. The presence of
        # capacity_config.attribute_capacities.column_mappings selects
        # attribute-aware vs. simple allocation downstream.
        'capacity_config': step_config.get('capacity_config', {}),
        # Subset configuration
        'subset_key': step_config.get('subset_key')
    }

    # Use the existing venue allocator function
    stats = _allocate_to_venue_type(
        venue_type=venue_type,
        allocation_config=allocation_config,
        population=population,
        venues=venues,
        household_distributor=household_distributor,
        scope_geo_unit_names=scope_geo_unit_names,
        scope_sgu_codes=scope_sgu_codes
    )

    return stats
