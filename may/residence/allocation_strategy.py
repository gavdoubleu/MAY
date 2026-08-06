"""
Unified allocation strategy executor.

Executes both household and venue allocations in a single YAML-defined sequence.
"""

import csv
import os
import hashlib
import logging
import operator
import yaml

import numpy as np
from typing import Callable, Dict, List, Optional, Set, Tuple

from .venue_allocator import _allocate_to_venue_type
from .household_distributor import HouseholdError
from .composition_pattern import CompositionPattern
from .pending_flush_buffer import PendingFlushBuffer
from .unit_mutation_plan import (UnitMutationCollector, apply_unit_plan,
                                 index_venues_by_id)
from may.parallel import GeoUnitSlice, parallel_map_in_order
from may.utils import path_resolver as pr

logger = logging.getLogger("allocation_strategy")

# The venue type per-unit flushing stages: households are what allocation
# creates, and what dominates peak memory
HOUSEHOLD_VENUE_TYPE = 'household'

# The world workers read over fork copy-on-write. Published by the parent for
# the duration of the loop and cleared after; never pickled, never written by
# a worker in any way the parent is expected to see.
_ACTIVE_LOOP_CONTEXT = None


class _LoopUnitContext:
    """Everything a loop unit's steps need, held for workers to read on fork.

    A worker is handed only its unit code; the world itself arrives through
    copy-on-write, which is the whole reason the fork start method is used
    (ADR-0003). Pickling any of this per unit would cost more than the
    allocation it enables.
    """

    __slots__ = ('population', 'venues', 'household_distributor',
                 'numbered_steps', 'loop_units', 'smallest_level',
                 'global_seed')

    def __init__(self, population, venues, household_distributor,
                 numbered_steps, loop_units, smallest_level, global_seed):
        self.population = population
        self.venues = venues
        self.household_distributor = household_distributor
        self.numbered_steps = numbered_steps
        self.loop_units = loop_units
        self.smallest_level = smallest_level
        self.global_seed = global_seed


def _run_one_geo_unit(unit_slice, rng):
    """Worker: run every allocation step for one loop unit.

    Returns `(plan, stats_by_step, target_pattern_tallies)` — everything the
    parent needs and nothing that holds a live object. The world this mutates
    is the worker's private copy and is discarded when it exits; the plan is
    the only thing that survives.

    The `rng` the pool passes is deliberately unused. This loop's per-unit
    stream has always been seeded from the unit code (`_stable_unit_digest`),
    not from the Slice, and it must keep being seeded that way or the
    restructure would move the world it is supposed to leave alone.
    """
    context = _ACTIVE_LOOP_CONTEXT
    unit_code = unit_slice.unit_codes[0]
    unit = context.loop_units[unit_code]
    household_distributor = context.household_distributor

    household_distributor.set_random_generator(np.random.default_rng(
        np.random.SeedSequence((context.global_seed,
                                _stable_unit_digest(unit_code)))))

    if unit.level == context.smallest_level:
        scope_sgu_codes = {unit_code}
    else:
        scope_sgu_codes = {descendant.name for descendant
                           in unit.get_descendants(level=context.smallest_level)}
    # The unit's whole subtree (any level) — venues may sit between the loop
    # level and the smallest level
    scope_geo_unit_names = {unit_code} | {descendant.name for descendant
                                          in unit.get_descendants()}
    household_distributor.set_geo_unit_scope(scope_sgu_codes)

    # Collected on every run, not just parallel ones: the diff's allowlist is
    # what turns an unknown mutation site into a loud single-process failure
    # instead of a parallel-only divergence.
    collector = UnitMutationCollector(context.venues, household_distributor,
                                      unit_code, scope_geo_unit_names)

    logger.info("=" * 60)
    logger.info(f"Geo unit {unit_code} "
                f"({len(scope_sgu_codes)} {context.smallest_level} unit(s))")
    logger.info("=" * 60)

    stats_by_step = {}
    for step_number, step_name, step_type, step_config in context.numbered_steps:
        logger.info("-" * 60)
        logger.info(f"[{unit_code}] Step {step_number}: {step_name} ({step_type})")
        logger.info("-" * 60)

        description = step_config.get('description')
        if description:
            logger.info(f"Description: {description}")
            logger.info("")

        if step_type == 'household':
            stats = _execute_household_step(step_config, household_distributor)
        elif step_type == 'venue':
            stats = _execute_venue_step(step_config, context.population,
                                        context.venues, household_distributor,
                                        scope_geo_unit_names, scope_sgu_codes)
        elif step_type == 'household_excess':
            stats = _execute_household_excess_step(step_config, household_distributor)
        elif step_type == 'household_overflow':
            stats = _execute_household_overflow_step(step_config, household_distributor)
        elif step_type == 'household_promotion':
            stats = _execute_household_promotion_step(step_config, household_distributor)

        stats_by_step[step_name] = stats
        logger.info("")

    return (collector.collect(), stats_by_step,
            household_distributor.drain_target_pattern_tallies())


def execute_allocation_strategy(population,
                                venues,
                                household_distributor,
                                strategy_file: str = "data/households/allocation_strategy.yaml",
                                export_debug_csv: bool = False,
                                random_seed: Optional[int] = None,
                                subset_flusher: Optional[Callable] = None,
                                plan_sink: Optional[Callable] = None,
                                workers: Optional[int] = 1,
                                max_tasks_per_child: Optional[int] = None):
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
            `memory_budget_mb`; called with whole units' household subsets
            whenever the run's resident set exceeds that budget, and once
            more at end of allocation for the residue.
        plan_sink: Callable handed each loop unit's `UnitMutationPlan` as the
            parent applies it, in the loop's own unit order. Diagnostic only —
            the world is already built by the time it is called.
        workers: Size of the process pool the loop units run across. None takes
            the SLURM allocation, then the machine's CPU count. Every worker
            count forks, one included: a unit's steps run against a throwaway
            copy of the world, so running them in the parent would apply each
            unit twice. That makes a one-worker run a true control for a
            many-worker one rather than a different code path.

            Defaults to one, not to the machine — widening the pool is opt-in
            until the footprint work sizes it. Each worker's copy of the world
            starts shared with the parent, but copy-on-write is undone by
            refcounting alone, so a pool costs roughly a world per worker at a
            scale where one world is already the constraint.
        max_tasks_per_child: Retires and reforks a worker after that many loop
            units. A worker is reused for every unit it draws for the life of
            the pool, and nothing resets what it accumulated in between (see
            `CONTEXT.md`'s "Worker world copy") — unlike the parent's own
            memory, that growth is invisible to `memory_budget_mb` and
            unbounded by it. None (the default) never retires a worker,
            matching today's behaviour.

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

    _validate_step_names_are_unique(steps, strategy_file)

    # A stale key is not ignorable here: the run would hold the whole
    # world's membership live to export — the blow-up the bound exists to
    # prevent — and report `calls=0` while doing it
    if 'flush_per_unit' in strategy:
        raise HouseholdError(
            f"flush_per_unit in {strategy_file} was removed; use "
            f"memory_budget_mb: 0 for the same per-unit cadence, "
            f"or omit the key to hold membership live until export"
        )

    # Superseded rather than renamed: a member count cannot be carried to a
    # machine with different memory, which is the whole reason it is gone
    if 'pending_flush_member_bound' in strategy:
        raise HouseholdError(
            f"pending_flush_member_bound in {strategy_file} was removed; "
            f"cadence is now driven by measured memory. Set memory_budget_mb "
            f"to the megabytes this run may hold, or 0 to stage at every "
            f"opportunity as a bound of 1 did"
        )

    # Megabytes of resident set the run may occupy before staging a batch.
    # Unset means no buffering is wired at all and membership stays live to
    # the terminal whole-world flush; see docs/adr/parallelising_adrs/0008.
    memory_budget_mb = strategy.get('memory_budget_mb')
    if memory_budget_mb is not None and (isinstance(memory_budget_mb, bool)
                                         or not isinstance(memory_budget_mb, (int, float))
                                         or memory_budget_mb < 0):
        raise HouseholdError(
            f"memory_budget_mb in {strategy_file} must be a non-negative "
            f"number of megabytes, not {memory_budget_mb!r}; omit the key "
            f"to hold every household's membership live until export"
        )
    if memory_budget_mb is not None and subset_flusher is None:
        raise HouseholdError(
            f"memory_budget_mb in {strategy_file} requires "
            f"serialization to be enabled: staged subsets are written beside "
            f"the canonical HDF5 file, which does not exist for this run"
        )

    # Present whenever staging is possible, so a run that deliberately never
    # flushes reports its zero — that run is the control a sweep is compared
    # against, and a missing row is not the same as a zero one
    pending_flush = (PendingFlushBuffer(subset_flusher, logger,
                                        budget_mb=memory_budget_mb or 0)
                     if subset_flusher is not None else None)

    # Cheap config checks above run first; the three below parse the pattern
    # vocabulary, read the mixture CSV off disk and scan venues respectively,
    # so a misconfigured run dies on a dict lookup, not after file I/O it
    # will never use.
    _resolve_pattern_selectors(
        steps,
        household_distributor.household_pattern_vocabulary,
        household_distributor.categories,
    )
    _validate_step_patterns(steps, household_distributor.household_pattern_vocabulary)
    _setup_structure_mixture(strategy.get('mixture'), steps, household_distributor)
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

    # The parent is the sole creator of Venues and the sole minter of their ids
    # (ADR-0003/0004), so it needs the two indices a plan is resolved through.
    # The population's own id index is reused rather than copied — at England
    # scale a second dict over every person is gigabytes for nothing. The venue
    # index is built once and never goes stale: a plan names only venues that
    # pre-date the loop, describing its own households rather than naming them.
    people_by_id = population.people_by_id
    venues_by_id = index_venues_by_id(venues)
    unit_slices = [GeoUnitSlice([unit_code]) for unit_code in loop_units]

    global _ACTIVE_LOOP_CONTEXT
    _ACTIVE_LOOP_CONTEXT = _LoopUnitContext(
        population=population, venues=venues,
        household_distributor=household_distributor,
        numbered_steps=numbered_steps, loop_units=loop_units,
        smallest_level=smallest_level, global_seed=global_seed)
    try:
        # Units finish in whatever order they finish in, but arrive here in the
        # loop's own order and never any other: household ids come from the
        # parent's counter as it applies, so the apply order *is* the world's id
        # layout (ADR-0010). The pool holds the ones that finished early, which
        # is why it can bound how far ahead of the apply they are allowed to run.
        applied_units = 0
        for _, outcome in parallel_map_in_order(
                _run_one_geo_unit, unit_slices, global_seed=global_seed,
                workers=workers, force_fork=True,
                max_tasks_per_child=max_tasks_per_child):
            unit_plan, stats_by_step, tallies = outcome
            applied_units += 1

            created_subset_keys = apply_unit_plan(
                unit_plan, venues, household_distributor,
                people_by_id, venues_by_id)
            household_distributor.merge_target_pattern_tallies(tallies)
            for step_number, step_name, step_type, _ in numbered_steps:
                accumulated = all_stats.setdefault(
                    step_name, {'type': step_type, 'step_number': step_number})
                _merge_step_stats(accumulated, stats_by_step[step_name])

            if plan_sink is not None:
                plan_sink(unit_plan)

            if memory_budget_mb is not None and created_subset_keys:
                # Offered, not flushed: whether this unit's membership is
                # written now or held for a later batch is the buffer's call
                member_count = unit_plan.created_household_member_count
                pending_flush.offer(HOUSEHOLD_VENUE_TYPE,
                                    created_subset_keys, member_count)
                logger.info(f"[{unit_plan.unit_code}] Offered "
                            f"{len(created_subset_keys)} household subsets "
                            f"({member_count:,} members) to the "
                            f"pending-flush buffer")

            if (applied_units % progress_interval == 0
                    or applied_units == total_units):
                percent_complete = (applied_units / total_units) * 100
                logger.info(f"Geo-unit progress: {applied_units}/{total_units} "
                            f"({percent_complete:.1f}%) - "
                            f"{len(household_distributor.allocated_people):,} people allocated")
    finally:
        _ACTIVE_LOOP_CONTEXT = None

    if pending_flush is not None:
        pending_flush.drain()
        pending_flush.log_summary()

    # Clear the scope so any later use of the distributor sees the whole world
    household_distributor.set_geo_unit_scope(None)

    # Target patterns are matched per unit, so only the union over every unit
    # distinguishes a typo from a pattern that is simply absent locally
    household_distributor.report_unmatched_target_patterns()

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
    remaining_unallocated = household_distributor.get_available_people_count()
    logger.info(f"  Remaining unallocated: {remaining_unallocated:,}")

    # The scope was cleared above, so this is the whole world — the one place
    # a full breakdown is worth an O(population) scan. The per-step version of
    # this report is DEBUG only, because it costs a scan per step per unit.
    if remaining_unallocated:
        leftovers_by_category = household_distributor.get_available_people_by_category()
        logger.info("  Remaining by category:")
        for category in household_distributor.categories:
            logger.info(f"    {category.name}: "
                        f"{leftovers_by_category.get(category.name, 0):,}")

    _check_per_unit_counts_against_the_world(all_stats, remaining_unallocated)

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


def _check_per_unit_counts_against_the_world(all_stats: Dict,
                                             remaining_unallocated: int) -> None:
    """Warn when the summed per-unit leftovers disagree with the exact count.

    Each unit reports what is left in its own scope, and the world's leftovers
    are those sums added up — but only while everything a unit allocates lies
    inside that unit's scope. Household and venue steps both allocate within
    the scope; `mark_people_as_allocated` does not have to. One O(population)
    comparison per run buys a loud complaint instead of a quietly wrong total.

    Only the run's final step is comparable: an earlier step's total predates
    whatever ran after it. A final step that reports no counts (a venue step)
    leaves nothing to compare, so the check stands down.
    """
    if not all_stats:
        return
    final_step = max(all_stats.values(), key=lambda stats: stats.get('step_number', 0))
    if 'total_people_remaining' not in final_step:
        return

    remaining_by_arithmetic = final_step['total_people_remaining']
    if remaining_by_arithmetic != remaining_unallocated:
        logger.warning(
            f"  Per-unit people counts do not add up to the world: units report "
            f"{remaining_by_arithmetic:,} unallocated, an exact scan finds "
            f"{remaining_unallocated:,}. Something was allocated outside the "
            f"scope of the unit that was running, so the per-step totals above "
            f"are unreliable.")


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


_SELECTOR_OPS = {'>=': operator.ge, '>': operator.gt, '==': operator.eq,
                 '<=': operator.le, '<': operator.lt}


def _resolve_pattern_selectors(steps: List[Dict], vocabulary: set, categories: List) -> None:
    """Resolve `patterns_where` selectors and enforce build-step disjointness.

    Runs once, before any allocation. A selector is a list of
    {category, operator, value} conditions evaluated against each vocabulary
    pattern's minimum counts; the matches are written back into the step as an
    explicit `patterns:` list, so the executor below needs no changes. Every
    resolved or hand-written build-step pattern is then claimed exactly once —
    a pattern claimed twice would double its census build quota, so overlap is
    an error, and a `patterns: null` step takes whatever remains unclaimed.
    """
    build_steps = [s for s in steps if s.get('type') == 'household']
    if not build_steps or not vocabulary:
        return
    name_to_idx = {cat.name: idx for idx, cat in enumerate(categories)}

    for step in build_steps:
        where = step.get('patterns_where')
        if where is None:
            continue
        step_name = step.get('name', 'household')
        if step.get('patterns') is not None:
            raise HouseholdError(
                f"Step '{step_name}': give either 'patterns' or 'patterns_where', not both."
            )
        conditions = []
        for cond in where:
            cat, op, value = cond.get('category'), cond.get('operator'), cond.get('value')
            if cat not in name_to_idx:
                raise HouseholdError(
                    f"Step '{step_name}': patterns_where category {cat!r} is not one of "
                    f"{sorted(name_to_idx)}."
                )
            if op not in _SELECTOR_OPS:
                raise HouseholdError(
                    f"Step '{step_name}': patterns_where operator {op!r} is not one of "
                    f"{sorted(_SELECTOR_OPS)}."
                )
            conditions.append((name_to_idx[cat], _SELECTOR_OPS[op], value))

        matched = []
        for pattern_str in vocabulary:
            pattern = CompositionPattern.from_string(pattern_str)
            if all(op(pattern.get_min_count(idx), value) for idx, op, value in conditions):
                matched.append(pattern_str)
        if not matched:
            raise HouseholdError(
                f"Step '{step_name}': patterns_where matched no pattern in households.csv."
            )
        step['patterns'] = sorted(matched)
        del step['patterns_where']
        logger.info(f"Step '{step_name}': patterns_where resolved to {len(matched)} patterns")

    # pattern -> {interpretation-or-None: step name}. A step without an
    # `interpretation` claims the pattern's WHOLE census count (key None),
    # which conflicts with any other claim; interpretation steps share a
    # pattern as long as their interpretations differ — each takes its
    # mixture quota of the count.
    claimed: Dict[str, Dict[Optional[str], str]] = {}
    for step in build_steps:
        patterns = step.get('patterns')
        if patterns is None:
            continue
        step_name = step.get('name', 'household')
        interp = step.get('interpretation')
        for p in patterns:
            name = p.get('pattern') if isinstance(p, dict) else p
            holders = claimed.setdefault(name, {})
            conflict = (holders.get(interp) if interp is not None and None not in holders
                        else next(iter(holders.values()), None) if interp is None
                        else holders.get(None) or holders.get(interp))
            if conflict:
                raise HouseholdError(
                    f"Pattern '{name}' is claimed by both '{conflict}' and "
                    f"'{step_name}' — a pattern's census count is a build quota, so "
                    f"steps must claim distinct patterns (or distinct interpretations "
                    f"of one pattern under a mixture)."
                )
            holders[interp] = step_name

    for step in build_steps:
        if step.get('patterns') is not None:
            continue
        step_name = step.get('name', 'household')
        remainder = sorted(vocabulary - set(claimed))
        if not remainder:
            raise HouseholdError(
                f"Step '{step_name}' (patterns: null) has no patterns left — "
                f"earlier build steps already claim the whole vocabulary."
            )
        step['patterns'] = remainder
        for name in remainder:
            claimed[name] = {step.get('interpretation'): step_name}
        logger.info(f"Step '{step_name}': patterns null -> {len(remainder)} remaining patterns")


def _setup_structure_mixture(mixture_cfg: Optional[Dict], steps: List[Dict],
                             household_distributor) -> None:
    """Load the structure-mixture table and validate interpretation claims.

    A census composition pattern is a marginal: different household structures
    (a couple, a parent with an adult child, unrelated people) project onto
    the same pattern. The mixture table gives, per geo unit, the measured
    share of each interpretation, and build steps claim one interpretation
    each — the quota split happens at build time.

    Entirely opt-in: no `mixture:` block means no behavior change, and using
    `interpretation:` on a step without the block is an error.
    """
    interp_steps = [s for s in steps
                    if s.get('type') == 'household' and s.get('interpretation') is not None]
    if mixture_cfg is None:
        if interp_steps:
            names = [s.get('name', 'household') for s in interp_steps]
            raise HouseholdError(
                f"Step(s) {names} use 'interpretation' but the strategy has no "
                f"'mixture:' block declaring the shares file."
            )
        household_distributor.structure_mixture = None
        return

    path = pr.resolve(mixture_cfg.get('file', ''))
    if not path or not os.path.exists(path):
        raise HouseholdError(f"mixture.file not found: {path!r}")
    geo_level = mixture_cfg.get('geo_level')
    if geo_level not in household_distributor.geography.levels:
        raise HouseholdError(
            f"mixture.geo_level {geo_level!r} is not one of the configured "
            f"geography levels {household_distributor.geography.levels}."
        )

    shares: Dict[tuple, Dict[str, float]] = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['geo_unit'], row['pattern'])
            shares.setdefault(key, {})[row['interpretation']] = float(row['share'])

    interps_by_pattern: Dict[str, set] = {}
    for (geo_unit, pattern), parts in shares.items():
        total = sum(parts.values())
        if abs(total - 1.0) > 0.02:
            raise HouseholdError(
                f"mixture shares for ({geo_unit}, '{pattern}') sum to {total:.4f}, not 1."
            )
        for interp in parts:  # normalise away rounding residue
            parts[interp] /= total
        interps_by_pattern.setdefault(pattern, set()).update(parts)

    # Every interpretation of a claimed pattern must be claimed by exactly one
    # step, or part of its census count would silently never be built.
    claimed_by_pattern: Dict[str, set] = {}
    for step in interp_steps:
        step_name = step.get('name', 'household')
        interp = step['interpretation']
        for p in (step.get('patterns') or []):
            name = p.get('pattern') if isinstance(p, dict) else p
            if name not in interps_by_pattern:
                raise HouseholdError(
                    f"Step '{step_name}' claims interpretation '{interp}' of pattern "
                    f"'{name}', which has no rows in {path}."
                )
            if interp not in interps_by_pattern[name]:
                raise HouseholdError(
                    f"Step '{step_name}': interpretation '{interp}' does not exist for "
                    f"pattern '{name}' in {path} (has {sorted(interps_by_pattern[name])})."
                )
            claimed_by_pattern.setdefault(name, set()).add(interp)
    for pattern, claimed in claimed_by_pattern.items():
        missing = interps_by_pattern[pattern] - claimed
        if missing:
            raise HouseholdError(
                f"Pattern '{pattern}' has unclaimed interpretation(s) {sorted(missing)} — "
                f"that share of its census count would never be built. Add a step "
                f"(rule-free is fine) claiming each interpretation."
            )
    unused = set(interps_by_pattern) - set(claimed_by_pattern)
    if unused:
        logger.warning(
            f"mixture file has rows for pattern(s) never claimed with an "
            f"'interpretation:' step (built whole, mixture ignored): {sorted(unused)}"
        )

    household_distributor.structure_mixture = {'geo_level': geo_level, 'shares': shares}
    logger.info(
        f"Structure mixture loaded: {len(interps_by_pattern)} patterns x "
        f"{len({g for g, _ in shares})} geo units at {geo_level}"
    )


def _validate_step_names_are_unique(steps: List[Dict], strategy_file: str) -> None:
    """Fail loud on two steps sharing a name.

    The name keys the accumulated per-step stats, so a repeat merges two
    steps into one entry: their people counts sum on top of the per-unit
    sum, the reported step number is the first occurrence's, and the
    end-of-run check against the world then warns about arithmetic the
    config broke rather than out-of-scope allocation.
    """
    seen = set()
    for step_number, step_config in enumerate(steps, 1):
        step_name = step_config.get('name', f'Step {step_number}')
        if step_name in seen:
            raise HouseholdError(
                f"Duplicate step name '{step_name}' in {strategy_file}; "
                f"step names key the per-step statistics, so each must be unique"
            )
        seen.add(step_name)


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
            demotion_rules=demotion_rules,
            interpretation=step_config.get('interpretation')
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
