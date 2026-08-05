"""One loop unit's mutations, collected as a compact plan the parent applies.

Phase 5 splits the geo-unit outer loop in two: a *body* that works out what a
unit's households should look like, and a *parent* that is the sole creator of
Venues and the sole minter of their ids (ADR-0003/0004). The body runs in a
forked worker from cycle 4 on, so nothing it mutates survives — it has to
describe its work instead, and the description has to cross a pickle boundary.
Hence "object-free": ids and plain values only, never live `Person` or `Venue`.

**Collector, not recorder.** The body is left exactly as it was — distributor,
promoter, excess handler and venue allocator all still mutate the world in
front of them. `UnitMutationCollector` snapshots the world before the unit and
diffs it after. The alternative, threading a plan object through those four
modules, is not buildable: `household_promoter` and `household_excess_handler`
read back households the distributor created earlier in the *same* unit, so
there has to be a real world for them to read. See ADR-0009.

**Fail loud.** The snapshot is of key *sets* (plus list lengths and scalar
values), never a deep copy, so it is cheap enough to run per unit. Every key it
sees change must have a declared shipping rule; anything else raises. A
mutation site nobody knew about therefore becomes a single-process crash rather
than a silent parallel-only divergence — the one failure mode that would be
near-impossible to diagnose later.

Nothing in the loop removes a member (`Subset.remove_member` has no caller in
`may/residence/`), so a unit's effect on the world is append-only and the diff
is exact.
"""
import logging
from itertools import islice
from typing import Dict, List, Optional, Set, Tuple

from .household_distributor import HouseholdError

logger = logging.getLogger("unit_mutation_plan")

HOUSEHOLD_VENUE_TYPE = 'household'

# Properties of a newly created household that are shipped in the plan.
SHIPPED_HOUSEHOLD_PROPERTY_KEYS = frozenset({'original_pattern',
                                             'allocation_pattern'})

# Properties the parent rebuilds rather than ships. `_age_categories` is the
# distributor's own category list — identical in every worker, and a list of
# live objects; the rest are stamped on by `VenueManager.create_venue` from the
# venue config, so the applier gets them for free by going through it.
REBUILT_HOUSEHOLD_PROPERTY_KEYS = frozenset({'_age_categories', 'is_residence',
                                             'subset_categories', 'subset_key'})

# The only person property the loop writes: `household_excess_handler` records
# a romantic pairing on both partners.
SHIPPED_PERSON_PROPERTY_KEYS = frozenset({'cohabiting_couple'})

# Values compared by identity of content in the venue-property snapshot.
_SCALAR_TYPES = (str, int, float, bool, type(None))

_ABSENT = object()
_OPAQUE = ('opaque',)


def _is_resident_slot(property_key: str) -> bool:
    """`venue_allocator` appends live people to a venue's `residents` list, and
    to a `residents_<column>` list per attribute slot. Those are the only
    pre-existing venue properties the loop is allowed to grow.
    """
    return property_key == 'residents' or property_key.startswith('residents_')


def _property_fingerprint(value):
    """Cheap comparable stand-in for a venue property value.

    Lists are compared by length — they are append-only resident lists, so a
    length change is exactly an append and the tail is what was appended.
    Scalars are compared by value. Anything else is opaque: a change in it
    cannot be seen, which is why the allowlist below is a *key* allowlist.
    """
    if isinstance(value, list):
        return ('list', len(value))
    if isinstance(value, _SCALAR_TYPES):
        return ('scalar', value)
    return _OPAQUE


class UnitMutationPlan:
    """Everything one loop unit did, in terms the parent can replay.

    Attributes:
        unit_code: the loop unit this plan came from.
        created_households: `(geo_unit_name, properties, subsets)` per household
            in creation order, where `subsets` is `(subset_key, member_ids)` in
            subset-index order. Creation order is what fixes venue ids, so the
            parent must apply plans in the loop's own unit order.
        existing_venue_members: `(venue_id, subset_key, person_ids)` for people
            put into a venue that pre-existed the loop (care homes, dorms).
        venue_property_appends: `(venue_id, slot_key, person_ids)` for the
            resident lists `venue_allocator` maintains alongside the subsets.
        person_property_updates: `person_id -> {key: value}`.
        newly_allocated_person_ids: ids added to `allocated_people`. Mostly
            derivable from the records above, but `mark_people_as_allocated`
            does not have to go through them, so it is shipped outright.
    """

    __slots__ = ('unit_code', 'created_households', 'existing_venue_members',
                 'venue_property_appends', 'person_property_updates',
                 'newly_allocated_person_ids')

    def __init__(self, unit_code, created_households, existing_venue_members,
                 venue_property_appends, person_property_updates,
                 newly_allocated_person_ids):
        self.unit_code = unit_code
        self.created_households = created_households
        self.existing_venue_members = existing_venue_members
        self.venue_property_appends = venue_property_appends
        self.person_property_updates = person_property_updates
        self.newly_allocated_person_ids = newly_allocated_person_ids

    @property
    def created_household_member_count(self) -> int:
        """Members across every household this unit created — what a flush of
        this unit's membership costs, and what deferring it holds in memory.
        """
        return sum(len(member_ids)
                   for _, _, subsets in self.created_households
                   for _, member_ids in subsets)

    def __repr__(self):
        return (f"<UnitMutationPlan {self.unit_code}: "
                f"{len(self.created_households)} households, "
                f"{self.created_household_member_count} members>")


class UnitMutationCollector:
    """Snapshot the world before a loop unit runs, diff it after.

    Construct it with the unit's scope already set on the distributor, run the
    unit's steps, then call `collect()`.
    """

    def __init__(self, venues, household_distributor, unit_code: str,
                 scope_geo_unit_names: Set[str]):
        self._venues = venues
        self._distributor = household_distributor
        self._geography = household_distributor.geography
        self._unit_code = unit_code
        self._scope_geo_unit_names = scope_geo_unit_names

        # Venues are appended to their type's dict, so a count now identifies
        # this unit's creations as that dict's tail afterwards.
        self._venue_counts_before = {
            venue_type: len(venues_by_id) for venue_type, venues_by_id
            in venues.venues_by_type_and_id.items()}

        self._existing_venues = self._venues_in_scope()
        self._subset_members_before: Dict[Tuple[int, object], Set[int]] = {}
        self._venue_properties_before: Dict[int, Dict[str, tuple]] = {}
        for venue in self._existing_venues:
            for subset_key, subset in venue.subsets.items():
                self._subset_members_before[(venue.id, subset_key)] = {
                    person.id for person in subset.members}
            self._venue_properties_before[venue.id] = {
                key: _property_fingerprint(value)
                for key, value in venue.properties.items()}

        # People are held by reference for the diff — the collector runs
        # worker-side, where the objects are real; only the plan crosses IPC.
        self._people_in_scope = list(household_distributor.people_in_scope())
        self._person_property_keys_before = {
            person.id: frozenset(person.properties)
            for person in self._people_in_scope}
        self._allocated_before = set(household_distributor.allocated_people)

    def _venues_in_scope(self) -> List:
        """Every venue already sitting in the unit's subtree, by id.

        Read off the geographical units rather than scanned out of the venue
        manager: the subtree holds the handful of care homes and dorms a venue
        step can touch, where the manager holds the whole world's households.
        """
        venues_by_id = {}
        for geo_unit_name in self._scope_geo_unit_names:
            geo_unit = self._geography.get_unit(geo_unit_name)
            if geo_unit is None:
                continue
            for venue in geo_unit.venues:
                venues_by_id[venue.id] = venue
        return [venues_by_id[venue_id] for venue_id in sorted(venues_by_id)]

    def collect(self) -> UnitMutationPlan:
        """Diff the world against the snapshot and return the unit's plan."""
        existing_members, property_appends = self._collect_existing_venue_growth()
        newly_allocated = sorted(set(self._distributor.allocated_people)
                                 - self._allocated_before)
        return UnitMutationPlan(
            unit_code=self._unit_code,
            created_households=self._collect_created_households(),
            existing_venue_members=existing_members,
            venue_property_appends=property_appends,
            person_property_updates=self._collect_person_property_updates(),
            newly_allocated_person_ids=newly_allocated,
        )

    def _collect_created_households(self) -> List[Tuple]:
        created = []
        for venue_type, venues_by_id in self._venues.venues_by_type_and_id.items():
            new_count = len(venues_by_id) - self._venue_counts_before.get(venue_type, 0)
            if new_count <= 0:
                continue
            if venue_type != HOUSEHOLD_VENUE_TYPE:
                raise HouseholdError(
                    f"Geo unit '{self._unit_code}' created {new_count} "
                    f"'{venue_type}' venue(s). Only households are created "
                    f"inside the loop; a plan cannot describe anything else, "
                    f"so the parent would silently not create them."
                )
            # Walk the tail backwards so the cost is in the new households,
            # not in the whole world's, then restore creation order.
            new_venues = list(islice(reversed(venues_by_id.values()), new_count))
            created.extend(self._describe_created_household(venue)
                           for venue in reversed(new_venues))
        return created

    def _describe_created_household(self, venue) -> Tuple:
        geo_unit_name = venue.geographical_unit.name
        if geo_unit_name not in self._scope_geo_unit_names:
            raise HouseholdError(
                f"Household venue {venue.id} sits in '{geo_unit_name}', "
                f"outside geo unit '{self._unit_code}', but was counted among "
                f"the households it created — the venue manager's household "
                f"ordering cannot be trusted to describe a unit's work."
            )
        unknown = (set(venue.properties) - SHIPPED_HOUSEHOLD_PROPERTY_KEYS
                   - REBUILT_HOUSEHOLD_PROPERTY_KEYS)
        if unknown:
            raise HouseholdError(
                f"Household venue {venue.id} in geo unit '{self._unit_code}' "
                f"carries propert{'y' if len(unknown) == 1 else 'ies'} "
                f"{sorted(unknown)} with no shipping rule. Add them to "
                f"SHIPPED_HOUSEHOLD_PROPERTY_KEYS (or to the rebuilt set if "
                f"the parent can derive them) — otherwise the parallel world "
                f"would quietly lose them."
            )
        properties = {key: venue.properties[key]
                      for key in SHIPPED_HOUSEHOLD_PROPERTY_KEYS
                      if key in venue.properties}
        subsets = [(subset_key, sorted(person.id for person in subset.members))
                   for subset_key, subset in self._subsets_in_index_order(venue)]
        return (geo_unit_name, properties, subsets)

    @staticmethod
    def _subsets_in_index_order(venue) -> List[Tuple]:
        """Subsets in the order they were first created.

        `add_to_subset` numbers a new subset by how many the venue already had,
        so replaying in this order reproduces every `subset_index` — which the
        HDF5 layout keys membership by.
        """
        return sorted(venue.subsets.items(),
                      key=lambda item: item[1].subset_index)

    def _collect_existing_venue_growth(self) -> Tuple[List, List]:
        member_records = []
        append_records = []
        for venue in self._existing_venues:
            for subset_key, subset in self._subsets_in_index_order(venue):
                members_before = self._subset_members_before.get(
                    (venue.id, subset_key), frozenset())
                added = sorted({person.id for person in subset.members}
                               - members_before)
                if added:
                    member_records.append((venue.id, subset_key, added))
            append_records.extend(self._collect_venue_property_appends(venue))
        return member_records, append_records

    def _collect_venue_property_appends(self, venue) -> List[Tuple]:
        appends = []
        properties_before = self._venue_properties_before[venue.id]
        for key, value in venue.properties.items():
            before = properties_before.get(key, _ABSENT)
            if before is not _ABSENT and before == _property_fingerprint(value):
                continue
            if not _is_resident_slot(key):
                raise HouseholdError(
                    f"Geo unit '{self._unit_code}' changed property '{key}' on "
                    f"pre-existing venue {venue.id} ({venue.type}), which has "
                    f"no shipping rule. A forked worker's change to it would "
                    f"never reach the parent's world."
                )
            appended_from = before[1] if before is not _ABSENT else 0
            appends.append((venue.id, key,
                            [person.id for person in value[appended_from:]]))
        return appends

    def _collect_person_property_updates(self) -> Dict[int, Dict]:
        updates = {}
        for person in self._people_in_scope:
            added = (set(person.properties)
                     - self._person_property_keys_before[person.id])
            if not added:
                continue
            unknown = added - SHIPPED_PERSON_PROPERTY_KEYS
            if unknown:
                raise HouseholdError(
                    f"Geo unit '{self._unit_code}' wrote person propert"
                    f"{'y' if len(unknown) == 1 else 'ies'} {sorted(unknown)} "
                    f"on person {person.id}, which have no shipping rule. Add "
                    f"them to SHIPPED_PERSON_PROPERTY_KEYS — otherwise the "
                    f"parallel world would quietly lose them."
                )
            updates[person.id] = {key: person.properties[key] for key in added}
        return updates


def index_venues_by_id(venues) -> Dict[int, object]:
    """`venue_id -> Venue` over the venues that exist before the loop starts.

    Built once, not per unit: a plan only ever names venues that pre-exist the
    loop (its own households are created by the applier, never named by id), so
    the index never goes stale.
    """
    return {venue.id: venue
            for venues_by_id in venues.venues_by_type_and_id.values()
            for venue in venues_by_id.values()}


def apply_unit_plan(plan: UnitMutationPlan, venues, household_distributor,
                    people_by_id: Dict[int, object],
                    venues_by_id: Dict[int, object]) -> None:
    """Replay one unit's plan into the parent's world.

    The parent is the sole Venue creator, so household ids come from its own
    counter here and nowhere else. Applying plans in the loop's unit order
    therefore reproduces the sequential world's id layout exactly.
    """
    geography = household_distributor.geography
    for geo_unit_name, properties, subsets in plan.created_households:
        household = venues.create_venue(
            venue_type=HOUSEHOLD_VENUE_TYPE,
            geo_unit=geography.get_unit(geo_unit_name),
            properties=dict(properties,
                            _age_categories=household_distributor.categories))
        for subset_key, member_ids in subsets:
            for person_id in member_ids:
                household.add_to_subset(people_by_id[person_id],
                                        subset_key=subset_key)

    for venue_id, subset_key, person_ids in plan.existing_venue_members:
        venue = venues_by_id[venue_id]
        for person_id in person_ids:
            venue.add_to_subset(people_by_id[person_id], subset_key=subset_key)

    for venue_id, slot_key, person_ids in plan.venue_property_appends:
        venue = venues_by_id[venue_id]
        venue.properties.setdefault(slot_key, []).extend(
            people_by_id[person_id] for person_id in person_ids)

    for person_id, properties in plan.person_property_updates.items():
        people_by_id[person_id].properties.update(properties)

    household_distributor.allocated_people.update(plan.newly_allocated_person_ids)
