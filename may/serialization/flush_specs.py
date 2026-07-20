"""Per-target dispatch table for `World.flush_properties` (see
docs/adr/0003, docs/adr/0004, docs/adr/0005). Each target's three points of
divergence — which serializer method to call, which live cohort to
tombstone, and which registry backs the tombstone — are captured once here
so `flush_properties` can run one uniform tail instead of three near-copies.
"""

from collections import namedtuple

from may.serialization.cohort_flush import CohortFlushTracker

FlushSpec = namedtuple(
    "FlushSpec", ["serializer_call", "cohort", "registry", "tracker", "configured"]
)

FLUSH_SPECS = {
    "population": FlushSpec(
        serializer_call=lambda world, venue_type, properties: (
            world._serializer.flush_population_properties(world, properties)
        ),
        cohort=lambda world, venue_type: world.population.people,
        registry=lambda world, venue_type: world._serializer._population_flush.flushed_names,
        tracker=lambda world, venue_type: world._serializer._population_flush,
        configured=lambda world, venue_type: world._serializer.config.get_person_properties(),
    ),
    "geo_units": FlushSpec(
        serializer_call=lambda world, venue_type, properties: (
            world._serializer.flush_geography_properties(world, properties)
        ),
        cohort=lambda world, venue_type: world.geography.units_by_id.values(),
        registry=lambda world, venue_type: world._serializer._geography_flush.flushed_names,
        tracker=lambda world, venue_type: world._serializer._geography_flush,
        configured=lambda world, venue_type: (
            world._serializer.config.get_geography_settings()['properties']
        ),
    ),
    "venues": FlushSpec(
        serializer_call=lambda world, venue_type, properties: (
            world._serializer.flush_venue_properties(world, venue_type, properties)
        ),
        cohort=lambda world, venue_type: world.venues.get_venues_by_type(venue_type),
        registry=lambda world, venue_type: (
            world._serializer._venue_flushes[venue_type].flushed_names
        ),
        tracker=lambda world, venue_type: world._serializer._venue_flushes.setdefault(
            venue_type, CohortFlushTracker(venue_type)
        ),
        configured=lambda world, venue_type: (
            world._serializer.config.get_venue_properties(venue_type)
        ),
    ),
}
