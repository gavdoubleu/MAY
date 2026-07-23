# Plan: parallelise the embarrassingly-parallel stages

> Source PRD: `tmp/prd/parallelise-easy-stuff.md`
> ADRs: `docs/adr/parallelising_adrs/0001`–`0005`

## Context

Two stages loop over independent units single-process today:
`attribute_assignment` (`may/attribute_assignment/assigner.py`, person-sliceable)
and `geo_unit_outer_loop` (`may/residence/allocation_strategy.py`, geo_unit-sliceable —
its geo-outer precondition work is already merged, see recent commits).
No existing multiprocessing/joblib infra. Goal: a thin process-based
`parallel_map(fn, list[Slice]) -> list[result]` each stage can reuse, with
each call site owning its own precondition (seeding, disjointness).

## Architectural decisions (durable, all phases)

- **Slicing rule (2×2)**: shared-capacity axis kills person-slicing;
  cross-geo spatial-query axis kills geo_unit-slicing. `attribute_assignment`
  = neither → person-slice; `geo_unit_outer_loop` = capacity only → geo_unit-slice.
  Read-only KDTree is *not* a blocker; cross-geo reach is.
- **Unit of parallelism**: OS processes (`multiprocessing`), not threads (GIL) — ADR-0001.
- **`Slice`**: stable seedable sub-partition of a Cohort (distinct from `Cohort`,
  CONTEXT.md). Geo Slices = lists of geo_unit codes; population Slices = sets of
  stable person-IDs (never `Person` objects or list positions).
- **Seeding**: per-Slice, from the Slice's own stable domain identity
  (`SeedSequence((global_seed, digest(unit_code)))` / sorted person-ID hash),
  never worker index or chunk position — ADR-0002. Output reproducible across
  worker count, not against pre-parallel history.
- **Mutation contract**: workers return mutation descriptions, never mutate
  `Person`/`Venue` in place (fork-COW makes worker writes invisible). Parent
  applies **streamed, per completed Slice**, not batched — preserves bounded
  peak memory (≈ `worker_count × per-Slice footprint`) — ADR-0003.
- **Household creation = plan, parent is sole Venue creator** — ADR-0004.
  Workers build provisional in-worker households (throwaway local IDs), return
  a compact **household plan**; parent mints real IDs via its counter,
  materialises in **canonical geo_unit-code order**, dispatches Slices from a
  **shared FIFO queue in geo_unit-code order** so completion tracks apply order.
- **Read-only shared state** (population arrays, spatial indices, `DataSource`
  tables): build once in parent before fork; fork-COW is free. Exception:
  `MultiKeyLookupSource` memoization caches — pre-warm or make per-worker.
- **TDD (red→green vertical slices)**: code-building phases (1, 2, 3, 5, 6) are
  built with the `tdd` skill — acceptance criteria written as failing tests
  first, one test → one implementation, never all-tests-then-all-code. Tests
  exercise public interfaces (`parallel_map`, assigner/allocation entry points,
  the comparator), not internals, so they survive the return-and-apply
  restructuring. Tier-1 bit-for-bit and Tier-2 shift checks are themselves the
  outer acceptance tests. **Phase 4 is excluded** — it is profiling/measurement,
  nothing to drive with a failing test.
- **Verification is two-tier** — ADR-0005:
  - **Tier-1**: bit-for-bit across worker count at a *fixed Slice partition*
    (1 worker vs N over the same Slices → identical world/digest). Mechanical
    correctness of return-and-apply plumbing. Cheap at two-LAD smoke scale.
  - **Tier-2**: statistical equivalence vs pre-parallel output (distributions
    unchanged, exact world not). Baseline is a **prerequisite deliverable**
    (Phase 1), built and captured before any parallel change lands.
- **Out of scope** (not deferred): `VenueDistributor`/`AllocationEngine` family
  (both axes blocked), commuting (4.7% of run; cheaper non-parallel win),
  mid-run resume (fail loud, rely on SLURM resubmission).

---

## Phase 1: Statistical-equivalence harness + captured baseline

**PRD needs**: Tier-2 comparison harness as prerequisite deliverable (ADR-0005 gate).

> Built with the `tdd` skill: acceptance criteria as failing tests first (red→green, one test → one impl).

**No parallel code lands in this phase.** It exists to make every later phase
comparable against a frozen pre-parallel reference.

### What to build

Comparison scripts computing the distributions that must be preserved from a
world's output: household-size distribution, company/venue age distributions
per geo_unit, per-attribute value proportions (the `attribute_assignment`
targets). A stored baseline artefact captured from the current (pre-parallel)
output, plus a comparator reporting per-metric statistical shift (with a
pass/fail significance verdict). Runs at the existing two-LAD smoke scale.

### Acceptance criteria

- [x] Harness extracts household-size, age-per-geo_unit (venue age → population age per geo, companies carry no age property), and attribute-proportion distributions — `may/verification/statistical_equivalence.py`
- [x] Baseline captured from current pre-parallel output and committed as a reproducible artefact — `verification_baselines/guildford_pre_parallel.json` (442 metrics; two-LAD canonical world TBC — see README)
- [x] Comparator reports per-metric shift + significance verdict (chi²/KS); self-comparison of baseline vs baseline passes cleanly
- [x] Comparator tdd'd: failing test first for a *known-shifted* distribution (must flag) and an *identical* one (must pass), before implementation — `tests/test_unit/may/verification/test_statistical_equivalence.py` (10 tests)
- [x] Documented as the gate every parallelisation phase must clear before merge — `verification_baselines/README.md`

---

## Phase 2: Thin `parallel_map` worker-pool + Slice/seeding substrate

**PRD needs**: shared thin `parallel_map`-shaped utility; per-Slice seeding; bounded pool.

> Built with the `tdd` skill: behaviours tested through the public `parallel_map`/`Slice` interface only (never pool internals), so tests survive the implementation. The Tier-1 self-test is the tracer bullet.

### What to build

A process-based `parallel_map(fn, slices) -> list[result]`: a bounded pool
pulls `Slice`s from a shared queue (worker count decoupled from Slice count).
`Slice` type + per-Slice seeding derived from stable domain identity
(geo_unit code / sorted person-IDs). Configurable pool size (env/config;
measured-footprint formula deferred to Phase 6). Not yet wired to a real stage
— proven with a trivial deterministic self-contained worker function.

### Acceptance criteria

- [ ] `parallel_map(fn, slices)` runs `fn` per Slice across a bounded process pool, returns results
- [ ] Per-Slice RNG seeded by domain identity; a Slice's stream is identical regardless of which worker runs it or when
- [ ] **Tier-1 self-test**: 1 worker vs N workers over the same Slices → identical results
- [ ] Pool size configurable; defaults sane when SLURM env absent
- [ ] Read-only shared state (numpy arrays) shared via fork-COW, not re-pickled per Slice

---

## Phase 3: `attribute_assignment` parallelised (first tracer bullet)

**PRD needs**: `attribute_assignment` person-sliced under the return-compact-array apply path.

*Quadrant: no capacity, no cross-geo → person-sliceable. PRD gate says "Proceed"
(both halves O(1); win from population size). No profiling gate needed.*

> Built with the `tdd` skill: restructure tested through the assigner *entry point* (public API), with Tier-1 bit-for-bit as the outer acceptance test written before the return-and-apply restructure.

### What to build

Introduce a geo_unit-keyed partition of `eligible_people` into ID-keyed
population Slices (household-level strategies keyed by household, which sits in
one geo_unit — safe). Each worker runs its Slice's assignment and returns a
compact numpy value array + its sorted person-ID array; the parent applies them
under the ADR-0003 streamed apply path. First end-to-end use of Phase 2's pool.

### Acceptance criteria

- [ ] `eligible_people` partitioned into ID-keyed Slices; person- and household-level strategies both handled
- [ ] Worker returns `(sorted_person_ids, values)` numpy arrays; parent applies to `person.properties`
- [ ] **Tier-1**: 1 vs N workers at fixed Slice partition → bit-for-bit identical assignments
- [ ] **Tier-2**: attribute proportions show no significant shift vs Phase 1 baseline
- [ ] Existing assigner tests green; IPC ships raw numpy buffers (no per-person tuple pickling)

---

## Phase 4: `geo_unit_outer_loop` go/no-go profiling gate

**PRD needs**: profile decide:apply split before implementing geo_unit slicing.

**Gates Phase 5.** Multiprocessing only speeds the decide half; a dominant apply
half means restructuring buys little.

### What to build

Instrument the existing sequential per-unit step sequence with a timer around
the mutation-writing lines vs everything else. Report the decide:apply ratio at
smoke and (if feasible) representative scale. Decide and record the "worth it"
threshold.

### Acceptance criteria

- [ ] Decide-time vs apply-time split measured for the per-unit step sequence
- [ ] Worth-it threshold stated; go/no-go recorded
- [ ] If no-go: Phase 5 halted and rationale documented; if go: measurement informs Phase 6 footprint

---

## Phase 5: `geo_unit_outer_loop` parallelised

**PRD needs**: household-plan contract, parent-sole-creator, canonical dispatch, streamed flush, in-loop venue steps.

*Quadrant: shared capacity, no cross-geo → geo_unit-sliceable. ADR-0003/0004.*

> Built with the `tdd` skill: restructure tested through the allocation *entry point* (public API), with Tier-1 bit-for-bit as the outer acceptance test written before the household-plan restructure.

### What to build

Each worker processes one geo_unit's full step sequence over provisional
in-worker households, returning a compact **household plan** (per household:
pattern + member person-IDs + drawn per-member fields). The parent — sole Venue
creator — mints IDs via its counter in **canonical geo_unit-code order**,
dispatching Slices from a **shared FIFO queue in geo_unit-code order** with a
roughly pool-sized reorder buffer. Retire tail-slicing
(`_new_household_subset_keys`): each worker returns its own new-household keys.
`flush_per_unit`'s single-writer HDF5 append becomes the parent's streamed-apply
step. In-loop `venue_allocator.py` steps (`boarding_school`, `care_home`,
`student_dorms`) ride along — geo-scoped, disjoint per unit, safe.

### Acceptance criteria

- [ ] Workers return household plans (no live Venue objects over IPC); parent is sole creator, mints unique IDs first time
- [ ] Apply in canonical geo_unit-code order; shared-FIFO canonical dispatch; reorder buffer holds only compact plans
- [ ] In-loop `venue_allocator` steps parallel-safe within their geo_unit slice
- [ ] `flush_per_unit` runs as parent streamed-apply; `_new_household_subset_keys` tail-slicing removed
- [ ] **Tier-1**: 1 vs N workers at fixed Slice partition → bit-for-bit identical world
- [ ] **Tier-2**: household-size + venue-age-per-geo_unit distributions unshifted vs Phase 1 baseline

---

## Phase 6: Measured worker-pool sizing formula + scale rollout

**PRD needs**: concrete pool-sizing formula from measured per-Slice footprint.

> Built with the `tdd` skill: the sizing formula is a pure function — failing test first for the `min(cpus, mem//footprint)` cases (incl. SLURM-absent fallback), before implementation.

### What to build

Replace Phase 2's configurable size with
`min(SLURM_CPUS_PER_TASK, mem_budget // measured_per_slice_footprint)`, deriving
`measured_per_slice_footprint` via the `tmp/england_memory_test` RSS-windowing
methodology. Tune worker count at representative scale; confirm the
peak-memory ≈ `W × per-unit footprint` trade behaves as predicted.

### Acceptance criteria

- [ ] Pool size computed from measured per-Slice footprint + SLURM CPU budget
- [ ] Footprint sourced from the RSS-windowing harness, documented
- [ ] Scale run confirms wall-clock win and bounded peak-memory trade; worker count exposed as the tunable dial

---

## Verification (overall)

- **Ordering gate**: Phase 1 baseline lands and is captured before any of
  Phases 3/5 merge (ADR-0005). Phase 4 gates Phase 5.
- **Per phase**: full pytest suite (`tests/test_unit`, `tests/test_integration`).
- **Tier-1** embedded in Phases 2, 3, 5 acceptance criteria.
- **Tier-2** run against the Phase 1 baseline after Phases 3 and 5.
- **TDD**: Phases 1, 2, 3, 5, 6 built with the `tdd` skill (red→green vertical
  slices, public-interface tests). Phase 4 is measurement — no test-drive.

## Open questions (from PRD, to resolve in-phase)

- Phase 4: exact profiling method + the decide:apply ratio that is "worth it".
- Phase 5: precise **household plan** schema (minimal fields to rebuild
  households + subsets + `member_metadata` + activity maps without live objects).
- Phase 5: whether the canonical-order reorder buffer needs an explicit depth
  cap / backpressure, or the pool-sized bound suffices.
- Phase 6: where `measured_per_slice_footprint` comes from (reuse
  `tmp/england_memory_test` RSS windowing?).
- `care_home_visits` descoped here; its `_build_population_arrays` vectorisation
  is an independent prerequisite, not part of this PRD.
