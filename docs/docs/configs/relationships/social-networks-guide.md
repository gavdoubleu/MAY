# Social Networks: a practical guide

This is a task-focused walkthrough for editing `configs/2021/relationships/social_networks.yaml` to change how simulated people form social contacts. It assumes you can edit YAML but don't know the underlying Python. For the full key-by-key reference see [social-networks.md](social-networks.md); this guide instead explains **why** you'd change things and **what happens** when you do.

Three things you control:

1. **Pool** — who a person is even allowed to become friends with.
2. **Average number of connections** — how many contacts each person ends up with.
3. **Clustering** — whether those contacts also know each other (tight friend groups) or are scattered strangers (loose acquaintance network).

---

## 1. The shipped config, as a starting point

`configs/2021/relationships/social_networks.yaml` currently defines three networks:

```yaml
networks:
  - name: activity_peers
    network_type: activity_peers
    pool_type: activity
    pool: { activity: primary_activity }
    algorithm: random
    mean_count: 3
    ...

  - name: geographic_local
    network_type: intra_geo_unit
    pool_type: geographic
    pool: { level: SGU }
    algorithm: random
    mean_count: 2
    ...

  - name: geographic_community
    network_type: intra_geo_unit
    pool_type: geographic
    pool: { level: MGU }
    algorithm: random
    mean_count: 1
    ...
```

**Important**: none of these three use clustering. They draw connections uniformly at random from the pool — there's no "your friends know each other" effect. If you only ever touch `mean_count` on these entries, clustering never enters the picture. To get a tuneable clustering coefficient you need one of the *other* two `network_type`s described in section 4.

---

## 2. Pools: who can connect to whom

A pool is the candidate group a person's contacts are drawn from. Nobody outside the pool can ever become a contact for that network entry. There are two ways to define a pool:

### `pool_type: activity`
Groups people by a shared venue — e.g. everyone at the same school, same workplace.

```yaml
pool_type: activity
pool:
  activity: primary_activity   # the activity_map key naming the venue
```

Use this for contacts formed "at a place" (school friends, co-workers).

### `pool_type: geographic`
Groups people by geographic unit. There are three nested levels, smallest to largest: `SGU` → `MGU` → `LGU`.

```yaml
pool_type: geographic
pool:
  level: SGU   # or MGU, LGU
```

Smaller level (`SGU`) = smaller, tighter pool (immediate neighbours). Larger level (`MGU`/`LGU`) = bigger, looser pool (wider community). This is why the shipped config has both a `geographic_local` (SGU) and `geographic_community` (MGU) entry — two pools of different breadth, each contributing a separate layer of contacts under the same `storage_key: friendships`.

For the clustering-capable network types (section 4) there's a third pool style — a **distance annulus** — which restricts the pool to people between a minimum and maximum distance in km, rather than a named geographic level.

**To add a new pool**: add a new entry under `networks:` with the `pool_type`/`pool` you want and a `storage_key`. If you reuse an existing `storage_key` (e.g. `friendships`), the new pool's contacts are merged in on top of the existing ones (deduplicated) — you don't need a new key unless you want to keep this contact type separate for later analysis.

---

## 3. Average number of connections (degree)

`mean_count` sets the average number of contacts a person gets from that network entry.

```yaml
mean_count: 3
```

To give a subset of people a different count (e.g. some highly-connected people), add `degree_variants`:

```yaml
mean_count: 3
degree_variants:
  - probability: 0.10   # 10% of people
    count: 6             # ...get 6 contacts instead of 3
```

Multiple `degree_variants` entries can be stacked; if a person matches more than one, the last-listed match wins. This is how you'd model, e.g., a small "socially hyperactive" minority.

For the clustering-capable network types (section 4), `mean_count` has a slightly different role: it sets the base number of ring connections before any rewiring happens (see below), so think of it as "connections before clustering is applied," not an exact final average — with `degree_variants` layered on top the same way.

---

## 4. Clustering: making friend groups actually know each other

### Which `network_type`s support it

Only three of the five `network_type`s have a clustering coefficient at all:

| `network_type` | Pool shape | Default `clustering_level` |
|---|---|---|
| `local_social_network` | one geographic level (`pool.level`) | 0.8 |
| `spatial_social_network` | distance annulus (`pool.min_km` to `pool.max_km`) | 0.9 |
| `bounded_distance` | radius from 0 to `pool.max_km` | 0.7 |

`activity_peers` and `intra_geo_unit` (the two types in the shipped config) never cluster — `clustering_level` has no effect on them.

### How it works, in plain terms

Within a pool, people are first arranged in a ring and each person is connected to their `mean_count` nearest neighbours in that ring — this alone would produce maximum clustering (tight, clique-like friend groups, like villages where everyone knows everyone). Then, some of those connections get randomly "rewired" to a random other person in the same pool instead. The more rewiring, the less clustered and more randomly-scattered the network becomes.

`clustering_level` controls how much rewiring happens:

- **`clustering_level: 1.0`** → no rewiring → maximum clustering → tight cliques, everyone's friends are also each other's friends, but few "bridges" between distant parts of the pool.
- **`clustering_level: 0.0`** → full rewiring → effectively random connections → your friends are unlikely to know each other, but you're more likely to have a far-flung acquaintance.
- **Values in between (the shipped defaults, 0.7–0.9)** are the realistic middle ground: mostly tight friend groups, plus a few long-range "weak tie" connections — this is what real social networks tend to look like.

> **Note on the reference doc**: [social-networks.md](social-networks.md) currently states this backwards (says `0.0` gives a regular lattice and `1.0` gives a random graph). The code (`clustered_graph.py`, `spatial_kernels.py`) computes rewiring probability as `1.0 - clustering_level`, so **higher `clustering_level` means more clustering**, as described above. Worth fixing the reference doc to match.

### Worked example A: switch geographic_local to a clustered network

Take the shipped `geographic_local` entry and turn it into a clustered one by changing `network_type` and adding `clustering_level`:

```yaml
- name: geographic_local
  network_type: local_social_network   # was: intra_geo_unit
  pool_type: geographic
  pool: { level: SGU }
  mean_count: 2
  clustering_level: 0.85                # new — high clustering, tight local groups
  storage_key: friendships
  constraints:
    - type: numerical_attribute_difference
      attribute: age
      max_difference: 10
```

Effect: instead of each person's SGU neighbours being drawn at random, people in the same small area now form tight local cliques (their local friends tend to also be friends with each other), with only occasional random links breaking the pattern.

### Worked example B: looser, more random local network

Same pool, but drop clustering to get a more "everyone's a stranger to each other's other friends" network:

```yaml
- name: geographic_local
  network_type: local_social_network
  pool_type: geographic
  pool: { level: SGU }
  mean_count: 2
  clustering_level: 0.2                 # low clustering — mostly random rewiring
  storage_key: friendships
```

Effect: still limited to the same small geographic pool, but contacts are close to randomly drawn from it — closer to the behaviour of the original `intra_geo_unit` type, but with a knob to dial in some residual clustering rather than none.

### Worked example C: a wider, distance-based clustered network

To add an entirely new network layer — e.g. "wider community ties, moderately clustered" — using a distance annulus instead of a named geo level:

```yaml
- name: regional_ties
  network_type: spatial_social_network
  pool_type: geographic
  pool:
    level: MGU
    min_km: 2.0     # exclude very close neighbours (already covered by geographic_local)
    max_km: 15.0    # up to 15km away
  mean_count: 2
  clustering_level: 0.6
  storage_key: friendships
```

Effect: adds a second, coarser layer of contacts drawn from a wider ring (2–15km out), moderately clustered, merged into the same `friendships` set as the other networks.

---

## 5. Quick checklist for changes

- **Want people to only be able to befriend those in a certain place/venue?** → change `pool_type`/`pool`.
- **Want more or fewer average contacts?** → change `mean_count`, or add `degree_variants` for a sub-population.
- **Want friend groups to actually know each other (cliquey) vs. scattered contacts (random)?** → use `network_type: local_social_network`, `spatial_social_network`, or `bounded_distance`, and tune `clustering_level` (1.0 = cliquey, 0.0 = random).
- **Changed `network_type` to a clustering one but nothing changed?** → check you didn't leave `clustering_level` unset expecting a default of "no clustering" — the defaults (0.7–0.9) already produce fairly high clustering.
- **Added a new network entry?** → make sure `storage_key` is either reused (`friendships`, to merge) or, if new, also added to `population.properties` in `serialization_config.yaml` so it appears in the output.
