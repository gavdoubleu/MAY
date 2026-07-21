# Plan: geo_unit-outer residence allocation

> Source PRD: `tmp/prd/geo-unit-outer-loop-residence-allocation.md`

## Context

`execute_allocation_strategy` (`may/residence/allocation_strategy.py:18`) runs steps-outer/geo_units-inner: peak memory holds every person pool + household subset for the whole world; household subsets can't flush until end-of-timeline (sole blocker: `_assign_all_people` iterates venue members); no unit of parallelism. Invert to geo_unit-outer: per unit, run full step sequence, optionally flush that unit's household subsets to staging. Replacement, not a mode — outputs statistically equivalent, not bit-identical.

## Architectural decisions (durable, all phases)

- **Config**: strategy YAML gains top-level `loop_level` (geography level name; validated against `geography.levels`; **default = largest configured level** when absent, so existing YAMLs keep near-old behaviour) and `flush_per_unit` (bool, default false; requires serialization enabled, hard error otherwise).
- **RNG**: injected `numpy.random.Generator` threaded through residence code; per-unit streams via `SeedSequence((global_seed, stable_digest(unit_code)))` — deterministic, platform-independent digest, not Python `hash()`. Results independent of unit iteration order.
- **Coarse venues**: venue steps whose venues sit above `loop_level` → validation error at strategy load (fail loud; revisit if real configs hit it).
- **Staging schema unchanged**: `staging/<venue_type>.h5` keeps dataset names/dtypes (`subsets/{venue_ids,subset_indices,subset_names,member_counts,members_flat}` + `membership_metadata/...`); datasets become resizable (`maxshape=(None,)`); staged order = flush-call order; final export re-indexes by key.
- **Flush tracker semantics**: `CohortFlushTracker` subset keys become cumulative across batches; duplicate keys across batches error; final-export check = flushed ∪ live == all keys, no overlap.
- **Husks stay fail-loud**: no read-through on `.members`; staged reads only via explicit API. CONTEXT.md tombstone guarantee preserved.
- **`max_allocations`**: deleted from code/configs/docs, no deprecation.
- **Stats**: per-step stats accumulate additively across units; single end-of-run summary in current format.

---

## Phase 1: `_assign_all_people` from `world.population`

**PRD needs**: `_assign_all_people` refactor (closure blocker + double-counting).

### What to build
`may/attribute_assignment/assigner.py:413-421` collects people via `world.population.get_all_people()` instead of iterating `venue_manager.get_all_venues_list()` members; plumb population through (currently takes only `venue_manager`).

### Acceptance criteria
- [ ] Population collected from `world.population`; no venue-membership iteration
- [ ] Unit test: person in population but no venue is assigned
- [ ] Regression test: world with gutted household subsets → assignment succeeds
- [ ] Existing assigner tests green

---

## Phase 2: Delete `max_allocations`

**PRD needs**: `max_allocations` removed outright.

### What to build
Remove from `may/residence/venue_allocator.py` (~116-120, enforcement 154/192), all configs (`configs/2021/`, `configs/example/`, others), docs. Warn loudly if key appears in a step config.

### Acceptance criteria
- [ ] `grep -r max_allocations` empty (code/configs/docs)
- [ ] `test_max_allocations_caps_total_placed` deleted; suite green
- [ ] Stale-key warning unit test

---

## Phase 3: Injectable RNG (single stream)

**PRD needs**: groundwork for per-geo_unit RNG streams.

### What to build
Replace global `np.random.*` in `may/residence/household_distributor.py` (~312, 979, 1509, 1536, 1544) and `venue_allocator.py` (329, 336) with an injected `Generator`, threaded from `execute_allocation_strategy` through distributor/round-distributor/allocator. One stream, seeded from global seed via `SeedSequence` (locks Phase-4 spawn shape).

### Acceptance criteria
- [ ] No module-level `np.random.` calls under `may/residence/`
- [ ] Determinism test: same seed → identical household/venue membership across two runs
- [ ] Integration suite green

---

## Phase 4: The inversion — geo_unit-outer executor

**PRD needs**: geo_unit-outer executor; replace-don't-add; loop level configurable; per-unit RNG; aggregated stats.

### What to build
Restructure `execute_allocation_strategy`: parse `loop_level`, validate patterns once up front, enumerate `geography.get_units_by_level(loop_level)`; per unit: derive per-unit `Generator`, run full step sequence restricted to that unit's SGUs (`unit.get_descendants(level=geography.levels[0])`), accumulate stats; single summary at end. Restriction mechanics: household pools/counts already keyed by SGU — build/refresh only current unit's SGUs; venue steps geo-local already (simple: per-SGU deques; attribute: `(column, venue_geo_unit)` pools) — restrict to unit's venues/people; excess/overflow/promotion scoped to unit's SGUs (main unknown — budget review here). `flush_per_unit: true` recognised but raises "not yet supported".

### Acceptance criteria
- [ ] `tests/test_integration/test_allocation_strategy_integration.py` passes with `loop_level` set
- [ ] Order-independence test: permuted unit iteration order → identical world
- [ ] Stats aggregate additively; summary format unchanged
- [ ] Unknown `loop_level` errors; coarse-venue validation error; `flush_per_unit: true` → clear not-yet-supported error

---

## Phase 5: Incremental subset staging

**PRD needs**: incremental subset staging (per-unit batches, cumulative tracking, union verify).

### What to build
`flush_venue_subsets` (`may/serialization/world_serializer.py:256`) accepts a caller-specified batch of subsets (disjoint key batches, callable N times per venue_type): append to resizable staging datasets; per-batch write-back verify over appended region; gut per subset after verify. Rework partial/mixed guard (309-322): already-gutted-in-*this-batch* errors; earlier-batch husks fine. `CohortFlushTracker` (`may/serialization/cohort_flush.py`): `record_subset_keys` accumulates; overlap errors; `check_subset_drift` at export becomes union check.

### Acceptance criteria
- [ ] 2+ batches then export ≡ single-batch flush of same subsets (content-equivalent)
- [ ] Repeat-batch idempotent skip; overlapping batch errors; gutted-in-batch errors
- [ ] Adapted: `tests/test_unit/may/test_cohort_flush.py`, subset-flush block of `tests/test_integration/test_timeline_serialization.py` (~893-1330)

---

## Phase 6: Staged-member read API

> Built with the `tdd` skill: acceptance criteria as failing tests first.

**PRD needs**: explicit staged-member read API.

### What to build
Serializer API: husk → staged member ids (reuse `_subset_staging_index`/`_staged_member_ids`, `world_serializer.py:417-455`); plus ids→`Person` resolver via population manager. `.members` fail-loud untouched. Update CONTEXT.md husk docs.

### Acceptance criteria
- [ ] Staged ids match pre-gut membership across multiple batches
- [ ] Never-flushed subset → clear error; `.members` still raises `FlushedPropertyError`
- [ ] Unit tests beside `test_flushing.py`; integration read-back mid-run

---

## Phase 7: Wire `flush_per_unit` end-to-end

> Built with the `tdd` skill: acceptance criteria as failing tests first.

**PRD needs**: config sketch (`flush_per_unit: true`); flush requires serialization.

### What to build
In the unit loop: after a unit's steps, collect its household subset keys, call Phase-5 batch flush (subsets only). Validate `flush_per_unit` requires serialization (`world._serializer is not None`, pattern at `create_world.py:352`); plumb serializer hook from call site `may/world.py:433`. Final export runs union verification.

### Acceptance criteria
- [ ] Integration (stress world, `flush_per_unit: true`): after unit N, its household subsets are husks with correct staged ids via Phase-6 API
- [ ] Exported HDF5 subset membership identical to `flush_per_unit: false` run, same seed
- [ ] Attribute assignment (Phase 1 path) works over gutted world
- [ ] `flush_per_unit: true` without serialization → config error; false/absent ≡ Phase 4

---

## Verification (overall)

- **Phases 6 and 7 are built with the `tdd` skill**: red-green-refactor, acceptance criteria written as failing tests first. Phases 4 and 5 were verified by throwaway scripts instead, at the user's direction, so their criteria carry no regression coverage — backfill those tests opportunistically when touching the same code.
- Full pytest suite per phase (`tests/test_unit`, `tests/test_integration`).
- After Phase 4: run `create_world.py` on the example/2021 config, compare summary stats to a pre-change run (statistical equivalence, not bit-identity).
- After Phase 7: full-flush 2021 config run with per-unit household flushing; confirm memory drop via existing DEBUG per-stage memory logging; final export verifies staged union.

## Notes

- Docs (allocation-strategy docs, CONTEXT.md, ADRs) updated within each phase touching their subject.
- Stray `may/serialization/#world_serializer.py#` emacs backup: delete opportunistically.
- Open question resolved: stats with ratios/averages — audit in Phase 4; sum counts, recompute derived values.
