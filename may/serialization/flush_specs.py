"""Per-target dispatch table for `World.flush_properties` (see
docs/adr/0003, docs/adr/0004, docs/adr/0005). Each target's three points of
divergence — which serializer method to call, which live cohort to
tombstone, and which registry backs the tombstone — are captured once here
so `flush_properties` can run one uniform tail instead of three near-copies.
"""

from collections import namedtuple

FlushSpec = namedtuple("FlushSpec", ["serializer_call", "cohort", "registry"])

FLUSH_SPECS = {
    "population": FlushSpec(
        serializer_call=lambda world, venue_type, properties: (
            world._serializer.flush_population_properties(world, properties)
        ),
        cohort=lambda world, venue_type: world.population.people,
        registry=lambda world, venue_type: world._flushed_person_property_names,
    ),
    "geo_units": FlushSpec(
        serializer_call=lambda world, venue_type, properties: (
            world._serializer.flush_geography_properties(world, properties)
        ),
        cohort=lambda world, venue_type: world.geography.units_by_id.values(),
        registry=lambda world, venue_type: world._flushed_geo_unit_property_names,
    ),
    "venues": FlushSpec(
        serializer_call=lambda world, venue_type, properties: (
            world._serializer.flush_venue_properties(world, venue_type, properties)
        ),
        cohort=lambda world, venue_type: [
            v for v in world.venues.get_all_venues_list() if v.type == venue_type
        ],
        registry=lambda world, venue_type: (
            world._flushed_venue_property_names_by_type.setdefault(venue_type, set())
        ),
    ),
}
