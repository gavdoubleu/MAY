# Tier-2 statistical-equivalence baselines

Reference distributions for changes that redraw the random stream (ADR-0005).
Per-unit seeding derives from the unit's stable domain identity, so any change
to which identity names a unit produces a *different exact world* by design.
That is accepted; what must be preserved is the **distributions**. These
baselines are what a redrawn world is compared against.

See `CONTEXT.md` in this directory for the terms used below.

## What this gate does and does not cover

Tier-2 answers one question: after a redraw, do the distributions still agree?

It is **not** the gate for changing worker count. A change that leaves the
stream intact must reproduce the world bit-for-bit, and is checked that way
(Tier-1, `h5diff -c`, 1 vs N workers). Reaching for Tier-2 there would accept
a real corruption as long as it stayed distributionally plausible. Use Tier-2
only where bit-for-bit equality is not expected to hold.

## Reading a result

The report separates the metric count from the **gate width**:

```
Tier-2 statistical equivalence: PASS (441 metrics: 5 compared, 436 tied)
```

The 436 tied metrics are `age_by_geo:<unit>`. Age and geographical unit are
core attributes loaded with the population; no in-scope stage writes either,
and residence allocation never moves a person between units. They agree in
every run by construction and carry no evidence. **The gate is 5 metrics
wide**: four attribute proportions and household size.

Width, not count, sets the false-positive budget. At α=0.05 a width of 5
expects ~0.25 spurious failures per run — so a single `SHIFTED` metric is
already several times the chance rate and deserves investigation, not a
dismissal as noise. Reasoning from 441 would license waving through roughly
22, which is how a real shift gets explained away.

`ethnicity` is the tightest of the five (p=0.09 on the reference comparison).
If one metric flips on a reseed it will most likely be that one; treat a lone
`ethnicity` failure as unproven until a second metric joins it.

A `MISSING` verdict means the metric left the map, **not** that the property
left the world — see `CONTEXT.md`. Look in the HDF5 before drawing conclusions
about configuration.

## Metrics

Harness: `may/verification/statistical_equivalence.py`.

- `household_size` — ordinal, household subsets only (KS).
- `attribute:<property>` — nominal proportion per person property (chi-square).
  Relational and combinatorial fields are dropped by a cardinality guard; they
  are not proportions.
- `age_by_geo:<unit>` — ordinal age histogram per geo unit (KS). Companies
  carry no age property, so the geo-scoped age quantity available is the
  population's own age composition per unit.

## Running it

```bash
python -m may.verification.statistical_equivalence compare <baseline.json> <candidate.h5>
#   exit 0 = PASS, exit 1 = FAIL (any SHIFTED or MISSING metric)
```

## Artefacts

- `guildford_pre_loop_level.json` — 441 metrics, captured from a Guildford
  world built with the coarser `loop_level: XLGU`, single worker, no memory
  budget. Its candidate is the same world at the committed `LGU`.

  **Scope of what it has shown.** Guildford is one unit at *both* levels, so
  the only difference between reference and candidate was the code seeding the
  stream. The pass therefore demonstrates **reseed stability** and nothing
  more. The structural consequences of iterating a coarser level — units that
  contain several finer ones, and the ordering and residue questions that come
  with them — are untested by it and need a geography large enough to divide.

  A reference is only usable against candidates built from the same commit.
  Comparing across unrelated drift mixes changes and produces failures that
  say nothing about the change under test.
