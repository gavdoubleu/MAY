# allocation_strategy.yaml

Defines the complete ordered sequence of allocation steps executed during world creation. Controls which people go into which households or communal venues, and in what order.

**Topic:** [Households](index.md)  
**Path:** `configs/2021/households/allocation_strategy.yaml`

---

## Keys

| Key | Description |
|---|---|
| `enabled` | Activates household allocation; set `false` to skip entirely |
| `loop_level` | Geography level the outer allocation loop runs over |
| `flush_per_unit` | Stage each unit's household membership to disk as it completes |
| `steps` | Ordered list of allocation steps; five step types are available |

---

## `enabled`

```yaml
enabled: true
```

Set to `false` to bypass all household and venue allocation. Agents remain unhoused.

---

## `loop_level`

```yaml
loop_level: MGU
```

The geography level of the outer loop: the full step sequence runs to
completion for one unit at that level before moving to the next, rather than
running each step across the whole world in turn. Must name a level from the
geography config. Defaults to the largest configured level, which keeps the
behaviour closest to a single whole-world pass.

A finer level means fewer people and households held in memory at once. Each
unit draws from its own random stream, derived from the global seed and the
unit's code, so results do not depend on the order the units are visited.

Venue steps whose venues sit above `loop_level` are rejected at load: such a
venue spans several loop units and cannot be filled by any one of them.

---

## `flush_per_unit`

```yaml
flush_per_unit: true
```

Once a unit's steps finish, write the household membership it created to
staging and free it from memory. Peak memory then holds roughly one unit's
households instead of the whole world's. The final export reads the staged
batches back, so the exported file is unchanged.

Requires serialization to be enabled — staging is written beside the
canonical `.h5` file, and a run without one raises rather than silently
skipping the flush. Defaults to `false`.

Flushed membership is gone from memory: a subset's `.members` raises
`FlushedPropertyError` afterwards. Anything needing it must ask the
serializer (`get_staged_member_ids`, `get_staged_members`). Attribute
assignment reads the population directly and is unaffected.

---

## `steps`

```yaml
steps:
  - name: "Two-Adult Families with Children"
    type: "household"
    ...
  - name: "Elderly to Care Homes"
    type: "venue"
    ...
```

Each entry in `steps` is executed in order, top to bottom. Earlier steps have first pick of the unallocated population pool. There are five step types.

---

### Step type: `household`

```yaml
- name: "Two-Adult Families with Children"
  type: "household"
  patterns:
    - ">=2 >=0 2 0"
    - pattern: ">=2 >=0 >=0 >=0"
      assumption: "2 0 1 1"
  rule: "Two-adult family with kids"
  max_households: null
  max_household_size: 10
  allocate_flexible: true
  refresh_pools: false
  enable_demotion: null
  demotion_rules:
    ">=2 >=0 1 0": "Single-adult family with kids"
```

Creates new households matching the listed `patterns`. Each pattern is either a plain string or a dict with an `assumption` — the assumed concrete shape used for sizing when a pattern contains open-ended `>=` slots. `rule` is the name of a rule from `relationship_rules.yaml`; when present, it enforces internal age-gap and pairing constraints on the household members. Omit `rule` for unconstrained allocation.

`max_households` caps the number of households created by this step; `null` means no limit. `max_household_size` sets a member ceiling per household. `allocate_flexible` — when `true` — randomly fills `>=` slots up to the size cap. `refresh_pools` rebuilds the unallocated-person pool before this step runs; use `true` after any `venue` step to exclude residents already placed. `enable_demotion` overrides the global demotion setting from `households_config.yaml` for this step only; `null` inherits the global value.

`demotion_rules` maps a demoted pattern string to a replacement rule name. When the engine demotes a pattern to the key (e.g. `">=2 >=0 1 0"`), it switches to the named rule for that household rather than the step's default `rule`.

---

### Step type: `venue`

```yaml
- name: "Elderly to Care Homes"
  type: "venue"
  venue_type: "care_home"
  allocation_mode: "attribute_aware"
  use_attribute_capacities: true
  subset_key: "resident"
  eligibility:
    - attribute: "age"
      min: 50
  strategy: "oldest_first"
```

Sends eligible agents to a communal residence venue type. `venue_type` must match a key in `venues_config.yaml`. `allocation_mode: "attribute_aware"` activates age/sex slot matching using the `capacity_config.attribute_capacities` defined for that venue type; set `use_attribute_capacities: true` alongside it. `subset_key` is the name assigned to residents within the venue.

`eligibility` is a list of attribute filters applied before slot matching; each filter specifies an `attribute` and optional `min`/`max` (for numerical) or `value` (for categorical). `strategy` controls selection order within the eligible pool: `"random"`, `"oldest_first"`, or `"youngest_first"`. Placement is capped by per-venue capacities.

---

### Step type: `household_excess`

```yaml
- name: "Add Excess Kids to Families"
  type: "household_excess"
  target_patterns:
    - ">=2 >=0 2 0"
  add_category: "Kids"
  rule: "Add young adults to existing family"
  constraints:
    - category_sum: ["Kids", "Young Adults"]
      max: 8
    - category: "Kids"
      max: 4
    - household_size: true
      max: 10
  add_distribution:
    type: "poisson"
    mean: 2
    min: 0
    max: 4
  max_per_household: null
  refresh_pools: false
```

Adds more members of a single `add_category` into existing households whose current pattern matches one of the `target_patterns`. The optional `rule` enforces relationship constraints between new arrivals and existing members.

`constraints` cap the household size after addition. Three forms are available: `category_sum` sums multiple categories and applies a `max`; `category` applies a `max` to a single category; `household_size: true` caps total membership.

`add_distribution` controls how many members are added per household: `"poisson"` uses a Poisson distribution with `mean` as lambda and `min`/`max` clamps; `"weighted"` uses an explicit `probabilities` dict mapping counts to weights; `"normal"` uses a normal distribution with `mean` and `std`. Set `add_distribution: null` to add as many as constraints allow. `max_per_household` is an additional hard cap per household, independent of constraints.

---

### Step type: `household_promotion`

```yaml
- name: "Promote households for specific categories"
  type: "household_promotion"
  promotion_rules:
    - source_pattern: "0 0 2 0"
      target_pattern: ">=0 >=0 2 0"
      accept_categories: ["Kids", "Young Adults"]
      max_to_add: 3
  target_categories:
    - "Young Adults"
    - "Adults"
  refresh_pools: false
```

Loosens existing household patterns so they can absorb new members. Two forms are available. `promotion_rules` specifies explicit source → target pattern transformations: households currently matching `source_pattern` have their pattern replaced with `target_pattern`, and then accept up to `max_to_add` new members from `accept_categories`. The `target_categories` form (no `promotion_rules`) promotes all households to accept the listed categories — used for final catch-all passes.

---

### Step type: `household_overflow`

```yaml
- name: "Overflow remaining Young Adults"
  type: "household_overflow"
  target_patterns:
    - "0 >=0 >=0 >=0"
    - "0 >=0 0 0"
  add_category: "Young Adults"
  pattern_bias:
    "0 >=0 >=0 >=0": 2.0
    "0 >=0 0 0": 5.0
  refresh_pools: false
```

Last-resort step that distributes all remaining members of `add_category` across existing households matching `target_patterns`, regardless of capacity. Use at the end of the pipeline to ensure nobody is left unhoused. `pattern_bias` assigns relative sampling weights to each target pattern; higher weight means more overflow members are directed there. Patterns not listed in `pattern_bias` receive equal weight of 1.0.
