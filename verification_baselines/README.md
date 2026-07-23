# Tier-2 statistical-equivalence baselines

Captured pre-parallel distribution baselines for the parallelising work
(ADR-0005). Per-Slice RNG seeding changes the *exact* world (accepted); what
must be preserved is the **distributions**. These baselines are the reference
every parallelisation phase is compared against.

## The gate

**Before any parallelisation change (Phases 3/5) merges**, its smoke-scale
world output must compare cleanly against the matching pre-parallel baseline
here. A non-zero exit / any `SHIFTED`/`MISSING` metric blocks the merge.

Harness: `may/verification/statistical_equivalence.py`. Metrics extracted:

- `household_size` — ordinal, household subsets only (KS).
- `attribute:<property>` — nominal proportion per person property; relational
  fields (friendships, cohabiting-couple IDs) dropped by a cardinality guard
  (chi-square).
- `age_by_geo:<unit>` — ordinal age histogram per geo unit (KS). Companies
  carry no age property, so the geo-scoped age quantity is the population's own
  age composition per unit.

## Commands

```bash
# capture a baseline from a pre-parallel world output
python -m may.verification.statistical_equivalence capture <world.h5> <baseline.json>

# compare a candidate (e.g. post-parallel) world against a baseline
python -m may.verification.statistical_equivalence compare <baseline.json> <candidate.h5>
#   exit 0 = PASS (no significant shift), exit 1 = FAIL
```

## Artefacts

- `guildford_pre_parallel.json` — captured from `output/2021/guildford.h5`
  (single-LAD real pre-parallel output; 442 metrics). Regenerate with the
  `capture` command above. **This is the current gate reference.**

**Deferred:** broadening to all available small-scale (≤5-LAD) 2021 worlds.
The LAD level is geography **level 1** (full-England `world_state.h5` has 331).
Qualifying worlds under `output/2021/`: `guildford` (1 LAD), `temp` /
`world_state_mybranch` (2 LADs). `stirling.h5` is an incomplete export (no
population group) — not usable. Each is a one-command `capture` when wanted.
