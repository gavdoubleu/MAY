"""
World serializer for exporting to HDF5 format.

Exports world state (geography, population, venues) to HDF5 file
for loading in C++ simulation engine.
"""

import logging
import os
import shutil
import h5py
import numpy as np
from datetime import datetime
from collections import defaultdict
from .serialization_config import SerializationConfig
from .cohort_flush import CohortFlushTracker

logger = logging.getLogger("world_serializer")


class WorldSerializer:
    """
    Serializes World object to HDF5 format for C++ consumption.

    Uses SerializationConfig to determine which properties to include.
    Exports data in Structure-of-Arrays (SoA) format for efficient C++ loading.
    """

    def __init__(self, config_file):
        """
        Initialize WorldSerializer.

        Args:
            config_file: Path to serialization YAML configuration
        """
        self.config = SerializationConfig(config_file)
        self.compression_settings = self.config.get_compression_settings()
        self.registries = {}  # Registry of string-to-int mappings
        # Path of the canonical file if export_core() already created it this
        # run; export() then reopens it in 'r+' instead of rewriting from
        # scratch. See docs/adr/0003.
        self._core_output_file = None
        # One CohortFlushTracker per flush target: records flushed property
        # names and the cohort's row-id sequence, re-verifying it on every
        # later flush call and at final export (see docs/adr/0003,
        # docs/adr/0005). venue_type -> tracker, created lazily on first
        # flush for that type.
        self._population_flush = CohortFlushTracker("population")
        self._geography_flush = CohortFlushTracker("geo_units")
        self._venue_flushes = {}
        # Cohort sort caches for repeated flush_*_properties calls on the
        # same target, reused across calls when cohort count is unchanged
        # (see docs/adr/0006). Independent of, and never shared with,
        # export()'s own self._people_sorted/self._geo_units_sorted, which
        # always recompute fresh per export() call.
        self._population_flush_sorted = None
        self._geography_flush_sorted = None
        self._venue_flush_sorted = {}

    def export_core(self, world, output_file):
        """
        Create the canonical HDF5 file early, right after population
        generation, with the core /geography and /population datasets whose
        row order is already final (geo-sorted; id/age/sex/geo_unit never
        mutate). Property datasets are NOT created here — their dtypes are
        inferred from values that may not exist yet; they are written by
        later flush steps or by the final export(). See docs/adr/0003.
        """
        logger.info("=" * 60)
        logger.info("Creating canonical HDF5 (core datasets)")
        logger.info("=" * 60)
        logger.info(f"Output file: {output_file}")

        with h5py.File(output_file, 'w') as f:
            logger.info("Serializing geography (core)...")
            self._write_geography_core(f, world)
            logger.info("Serializing population (core)...")
            self._write_population_core(f, world)

        self._core_output_file = output_file
        logger.info("Canonical file created (core datasets written)")

    def _flush_cohort_properties(self, tracker, group_path, cohort_sorted,
                                  configured_properties, property_names,
                                  flushed_by_step_label, log_label):
        """
        Shared body of flush_{population,geography,venue}_properties():
        validate, record the cohort's row-id sequence against the tracker,
        write one dataset per property under `group_path` (require_group'd
        in sequence) in `cohort_sorted` order, stamp each with a
        'flushed_by_step' attr, then mark flushed on `tracker` only those
        properties that actually got a dataset written. Callers own the
        canonical-file check, the target-specific cohort sort and the
        configured-properties lookup — this only knows how to write a
        cohort's flush once those are settled.

        A property with every live value `None` at flush time gets no
        dataset (see `_write_property_array`) and is therefore left off
        `tracker`'s flushed set: it stays live in memory, un-tombstoned,
        and may be requested again by a later flush_properties call or
        picked up at final export once it has real values. Returns the
        list of property names actually written, so callers know what to
        tombstone.
        """
        tracker.validate(property_names, configured_properties)

        # Cohort must match any earlier flush: columns from both steps share
        # one row order (see docs/adr/0003).
        cohort_ids = [obj.id for obj in cohort_sorted]
        tracker.record_flush(cohort_ids)

        written_names = []
        with h5py.File(self._core_output_file, 'r+') as f:
            group = f
            for name in group_path:
                group = group.require_group(name)
            for prop_name in property_names:
                # A retry after a mid-loop failure re-enters from the first
                # property, so an earlier attempt may already have written
                # this dataset; drop it first so h5py's create_dataset does
                # not raise on the name collision.
                if prop_name in group:
                    del group[prop_name]
                self._write_property_array(group, prop_name, cohort_sorted)
                if prop_name in group:  # absent if every value was None
                    group[prop_name].attrs['flushed_by_step'] = flushed_by_step_label
                    written_names.append(prop_name)
                    logger.info(f"  Flushed {log_label} property '{prop_name}'")
                else:
                    logger.warning(
                        f"  Deferred {log_label} property '{prop_name}': "
                        f"all values were None, nothing written; not marked "
                        f"flushed, still live for a later flush/export."
                    )

        tracker.mark_flushed(written_names)
        return written_names

    def flush_population_properties(self, world, property_names):
        """
        Write person property columns to the canonical file mid-timeline.

        Reopens the canonical file in 'r+' and writes one dataset per
        property under /population/properties, in the frozen geo-sorted row
        order. Datasets are created lazily here — dtype inferred from live
        values (see docs/adr/0003). Each written dataset is stamped with a
        'flushed_by_step' attr for inspection; the authoritative skip list
        for the final export is self._population_flush.flushed_names. A
        requested property with every value `None` is deferred rather than
        flushed — see _flush_cohort_properties. Returns the property names
        actually written.

        The cohort sort is cached in self._population_flush_sorted and
        reused across repeated calls when the population count is
        unchanged, rather than re-sorted every call — see docs/adr/0006.
        """
        if self._core_output_file is None:
            raise RuntimeError(
                "flush_population_properties() requires the canonical file: "
                "call export_core() first."
            )
        configured = self.config.get_person_properties()
        people = world.population.people
        self._population_flush_sorted = self._cached_or_resorted(
            self._population_flush_sorted, len(people),
            lambda: self._stable_sort_by_geo_unit(people)[0],
        )
        return self._flush_cohort_properties(
            self._population_flush, ['population', 'properties'], self._population_flush_sorted,
            configured, property_names, 'flush_properties(target=population)',
            log_label='population',
        )

    def flush_geography_properties(self, world, property_names):
        """
        Write geographical unit property columns to the canonical file
        mid-timeline. Mirrors flush_population_properties(): lazy datasets
        under /geography/properties in the geo-sorted row order used by
        _write_geography_core(), stamped with a 'flushed_by_step' attr; the
        authoritative skip list for the final export is
        self._geography_flush.flushed_names. A requested property with
        every value `None` is deferred rather than flushed — see
        _flush_cohort_properties. Returns the property names actually
        written.

        The cohort sort is cached in self._geography_flush_sorted and
        reused across repeated calls when the unit count is unchanged,
        rather than re-sorted every call — see docs/adr/0006.
        """
        if self._core_output_file is None:
            raise RuntimeError(
                "flush_geography_properties() requires the canonical file: "
                "call export_core() first."
            )
        configured = self.config.get_geography_settings()['properties']
        units_by_id = world.geography.units_by_id
        self._geography_flush_sorted = self._cached_or_resorted(
            self._geography_flush_sorted, len(units_by_id),
            lambda: self._stable_sort_by_geo_unit(
                list(units_by_id.values()), geo_unit_key=lambda u: u
            )[0],
        )
        return self._flush_cohort_properties(
            self._geography_flush, ['geography', 'properties'], self._geography_flush_sorted,
            configured, property_names, 'flush_properties(target=geo_units)',
            log_label='geography',
        )

    def flush_venue_properties(self, world, venue_type, property_names):
        """
        Write one venue type's property columns to the canonical file
        mid-timeline. Rows follow the cohort order: venues of the type in
        the stable geo-sorted order _write_venues() will use, so columns
        align with ranks_in_type. The cohort must be closed — its Venue.id
        sequence is recorded here and re-verified at final export, which
        fails loud if venues of the type were added, removed or reordered
        after the flush. A requested property with every value `None` is
        deferred rather than flushed — see _flush_cohort_properties.
        Returns the property names actually written.

        The cohort sort is cached per venue_type in
        self._venue_flush_sorted and reused across repeated calls for the
        same type when its venue count is unchanged, rather than re-sorted
        every call — see docs/adr/0006.
        """
        if self._core_output_file is None:
            raise RuntimeError(
                "flush_venue_properties() requires the canonical file: "
                "call export_core() first."
            )
        configured = self.config.get_venue_properties(venue_type)
        tracker = self._venue_flushes.setdefault(venue_type, CohortFlushTracker(venue_type))

        # Cohort order: within-type slice of the stable geo-sort applied to
        # the full venue list by _write_venues().
        venues_of_type = world.venues.get_venues_by_type(venue_type)
        cached = self._venue_flush_sorted.get(venue_type)
        cohort_sorted = self._cached_or_resorted(
            cached, len(venues_of_type),
            lambda: self._stable_sort_by_geo_unit(venues_of_type)[0],
        )
        self._venue_flush_sorted[venue_type] = cohort_sorted
        return self._flush_cohort_properties(
            tracker, ['venues', 'properties', venue_type], cohort_sorted,
            configured, property_names,
            f'flush_properties(target=venues, venue_type={venue_type})',
            log_label=venue_type,
        )

    def _staging_dir(self):
        return os.path.join(os.path.dirname(self._core_output_file), 'staging')

    def _staging_path(self, venue_type):
        return os.path.join(self._staging_dir(), f'{venue_type}.h5')

    def flush_venue_subsets(self, world, venue_type):
        """
        Stage one venue type's subset membership to
        `<staging_dir>/<venue_type>.h5` and gut the live subsets, freeing
        their `.members`/`.member_metadata` (see may/flushing.py,
        docs/adr/0002). The cohort's `(Venue.id, subset_index)` keys are
        recorded on this venue_type's CohortFlushTracker and re-verified
        at final export (docs/adr/0005): no live subset may exist on this
        venue_type afterwards, and every staged subset must still be
        present as a husk.

        Staging is skipped, with a warning, if the venue_type currently
        has no subsets at all (mirrors the all-None property policy).
        """
        from may.flushing import gut_subset

        if self._core_output_file is None:
            raise RuntimeError(
                "flush_venue_subsets() requires the canonical file: "
                "call export_core() first."
            )

        tracker = self._venue_flushes.setdefault(venue_type, CohortFlushTracker(venue_type))

        venue_subset_pairs = [
            (venue, subset)
            for venue in world.venues.get_venues_by_type(venue_type)
            for subset in venue.subsets.values()
        ]
        if not venue_subset_pairs:
            logger.warning(
                f"No subsets to flush for venue_type {venue_type!r}; skipping."
            )
            return

        staging_dir = self._staging_dir()
        os.makedirs(staging_dir, exist_ok=True)
        staging_path = self._staging_path(venue_type)

        subset_keys = [(venue.id, subset.subset_index) for venue, subset in venue_subset_pairs]
        subset_names = [subset.subset_name for _, subset in venue_subset_pairs]
        member_lists = [[person.id for person in subset.members] for _, subset in venue_subset_pairs]
        member_counts = [len(members) for members in member_lists]
        total_members = sum(member_counts)

        with h5py.File(staging_path, 'w') as f:
            subsets_group = f.create_group('subsets')
            self._create_dataset(
                subsets_group, 'venue_ids',
                np.array([key[0] for key in subset_keys], dtype=np.int32),
            )
            self._create_dataset(
                subsets_group, 'subset_indices',
                np.array([key[1] for key in subset_keys], dtype=np.int32),
            )
            self._create_dataset(
                subsets_group, 'subset_names',
                np.array(subset_names, dtype=h5py.string_dtype()),
            )
            self._create_dataset(
                subsets_group, 'member_counts',
                np.array(member_counts, dtype=np.int32),
            )
            members_flat = np.array(
                [person_id for members in member_lists for person_id in members],
                dtype=np.int32,
            )
            self._create_dataset(subsets_group, 'members_flat', members_flat)

            # Empty-but-present membership_metadata shape: households carry
            # no member_metadata (Phase B tracer), so the schema exists but
            # no per-subset retrieval index is built here — see Phase B plan.
            meta_group = f.create_group('membership_metadata')
            self._create_dataset(meta_group, 'person_ids', np.empty(0, dtype=np.int32))
            self._create_dataset(meta_group, 'venue_ids', np.empty(0, dtype=np.int32))
            self._create_dataset(
                meta_group, 'field_names', np.empty(0, dtype=h5py.string_dtype()),
            )

        with h5py.File(staging_path, 'r') as f:
            written_total = f['subsets/members_flat'].shape[0]
        if written_total != total_members:
            raise RuntimeError(
                f"Staging write for venue_type {venue_type!r} verification "
                f"failed: expected {total_members} staged members, found "
                f"{written_total}."
            )

        for (venue, subset), member_count in zip(venue_subset_pairs, member_counts):
            gut_subset(subset, member_count)

        tracker.record_subset_keys(subset_keys)
        logger.info(
            f"  Flushed {venue_type} subsets to staging: "
            f"{len(venue_subset_pairs)} subsets, {total_members} memberships"
        )

    def _subset_staging_index(self, venue_type):
        """Lazily open the staging file for `venue_type` (if this run
        flushed its subsets) and build an in-memory
        {(venue_id, subset_index): (start, count)} index into its
        `members_flat` array from the small metadata arrays. Cached in
        self._subset_staging_readers so repeated husks reuse one open
        reader and one index."""
        if venue_type in self._subset_staging_readers:
            return self._subset_staging_readers[venue_type]

        reader = h5py.File(self._staging_path(venue_type), 'r')
        venue_ids = reader['subsets/venue_ids'][:]
        subset_indices = reader['subsets/subset_indices'][:]
        member_counts = reader['subsets/member_counts'][:]

        index = {}
        start = 0
        for venue_id, subset_index, count in zip(venue_ids, subset_indices, member_counts):
            index[(int(venue_id), int(subset_index))] = (start, int(count))
            start += int(count)

        self._subset_staging_readers[venue_type] = (reader, index)
        return reader, index

    def _staged_member_ids(self, subset):
        """Read a gutted subset's member ids back from staging."""
        reader, index = self._subset_staging_index(subset.venue.type)
        start, count = index[(subset.venue.id, subset.subset_index)]
        return reader['subsets/members_flat'][start:start + count].tolist()

    def _close_subset_staging_readers(self):
        for reader, _ in self._subset_staging_readers.values():
            reader.close()
        self._subset_staging_readers = {}

    def export(self, world, output_file):
        """
        Export world to HDF5 file.

        If export_core() already created the canonical file this run, the
        file is reopened in 'r+' and the core /geography and /population
        datasets are not rewritten.
        """
        # Internal state for export coordination
        self._venue_to_global_id = {}
        self._venue_type_id_map = {}
        self._people_sorted = None
        self._geo_units_sorted = None
        self._subset_staging_readers = {}
        logger.info("=" * 60)
        logger.info("Exporting World to HDF5")
        logger.info("=" * 60)
        logger.info(f"Output file: {output_file}")

        core_written = self._core_output_file is not None
        if core_written and self._core_output_file != output_file:
            raise ValueError(
                f"export() target {output_file!r} differs from the canonical "
                f"file created by export_core() ({self._core_output_file!r})."
            )

        stats = {
            'num_people': 0,
            'num_venues': 0,
            'num_geo_units': 0,
            'num_subsets': 0,
        }

        with h5py.File(output_file, 'r+' if core_written else 'w') as f:
            # Write metadata
            self._write_metadata(f, world, stats)

            # Write geography
            logger.info("Serializing geography...")
            if core_written:
                # Re-derive the frozen row order (stable sort of immutable
                # geo_unit ids — identical to the export_core() order).
                self._geo_units_sorted, _ = self._stable_sort_by_geo_unit(
                    list(world.geography.units_by_id.values()), geo_unit_key=lambda u: u
                )
            else:
                self._write_geography_core(f, world)
            self._write_geography_properties(f, world)
            stats['num_geo_units'] = len(world.geography.get_all_units())

            # Write population
            logger.info("Serializing population...")
            if core_written:
                # Re-derive the frozen row order (stable sort of immutable
                # geo_unit ids — identical to the export_core() order).
                self._people_sorted, _ = self._stable_sort_by_geo_unit(world.population.people)
            else:
                self._write_population_core(f, world)
            self._write_population_properties(f, world)
            stats['num_people'] = len(world.population.people)

            # Write venues
            logger.info("Serializing venues...")
            stats['num_subsets'] = self._write_venues(f, world)
            stats['num_venues'] = sum(len(d) for d in world.venues.venues_by_type_and_id.values())

            # Write activity mappings
            logger.info("Serializing activity mappings...")
            self._write_activity_mappings(f, world)

            # Write registries (Enum mappings for C++)
            logger.info("Writing string registries...")
            self._write_registries(f)

        # Any exception above propagates before this point, so staging/
        # survives a crash automatically for inspection/retry.
        self._close_subset_staging_readers()
        if self._core_output_file is not None:
            staging_dir = self._staging_dir()
            if os.path.isdir(staging_dir):
                try:
                    shutil.rmtree(staging_dir)
                except OSError as error:
                    logger.warning(f"Could not remove staging dir {staging_dir!r}: {error}")

        logger.info("")
        logger.info("Export complete!")
        logger.info(f"  Geography units: {stats['num_geo_units']:,}")
        logger.info(f"  People: {stats['num_people']:,}")
        logger.info(f"  Venues: {stats['num_venues']:,}")
        logger.info(f"  Subsets: {stats['num_subsets']:,}")
        logger.info("=" * 60)

        return stats

    def _write_metadata(self, f, world, stats):
        """Write metadata attributes to root of HDF5 file."""
        metadata_settings = self.config.get_metadata_settings()

        if not metadata_settings['include']:
            return

        logger.info("Writing metadata...")

        # Always include counts
        f.attrs['num_people'] = len(world.population.people)
        f.attrs['num_venues'] = sum(len(d) for d in world.venues.venues_by_type_and_id.values())
        f.attrs['num_geo_units'] = len(world.geography.get_all_units())

        # Optional metadata fields
        metadata_fields = metadata_settings['fields']

        if 'creation_timestamp' in metadata_fields:
            f.attrs['creation_timestamp'] = datetime.now().isoformat()

        # Version info
        f.attrs['serialization_version'] = '1.0'
        f.attrs['MAY_version'] = '0.1.0'

    def _write_geography_core(self, f, world):
        """Write geography hierarchy core datasets (no properties) to HDF5."""
        geo_group = f.create_group('geography')
        geo_settings = self.config.get_geography_settings()

        # Get all units keyed by unique ID (units_by_id): a single name can
        # occur at multiple levels (e.g. SGU "DURHAM" parish under MGU
        # "DURHAM" county), so keying by ID captures every unit and all
        # hierarchy levels in the export.
        units_list, _ = self._stable_sort_by_geo_unit(
            list(world.geography.units_by_id.values()), geo_unit_key=lambda u: u
        )

        # Store sorted units for property serialization within this
        # export() call (mirrors self._people_sorted for population).
        self._geo_units_sorted = units_list

        if not units_list:
            logger.warning("No geographical units to serialize")
            return

        num_units = len(units_list)

        # Create ID → index mapping for efficient lookup
        id_to_index = {unit.id: idx for idx, unit in enumerate(units_list)}

        # Core attributes (always included)
        ids = np.array([unit.id for unit in units_list], dtype=np.int32)
        
        # Move geography names to metadata
        names = np.array([unit.name for unit in units_list], dtype=h5py.string_dtype())
        metadata_group = f.require_group('metadata/names')
        self._create_dataset(metadata_group, 'geography', names)

        # Intern geography levels — order by tree depth (root=0, leaves=highest)
        # Find depth of each level by walking parent chains
        level_depths = {}
        for unit in units_list:
            depth = 0
            ancestor = unit
            while ancestor.parent is not None:
                depth += 1
                ancestor = ancestor.parent
            level_depths[unit.level] = depth
        unique_levels = sorted(level_depths.keys(), key=lambda l: level_depths[l])
        level_to_id = {l: i for i, l in enumerate(unique_levels)}
        levels = np.array([level_to_id[unit.level] for unit in units_list], dtype=np.uint8)
        self.registries['geo_levels'] = level_to_id

        # Parent IDs (-1 for root units)
        parent_ids = np.array(
            [unit.parent.id if unit.parent else -1 for unit in units_list],
            dtype=np.int32
        )

        # Write core datasets
        self._create_dataset(geo_group, 'ids', ids)
        self._create_dataset(geo_group, 'levels', levels)
        self._create_dataset(geo_group, 'parent_ids', parent_ids)

        # Coordinates (optional)
        if geo_settings['include_coordinates']:
            latitudes = np.array(
                [unit.coordinates[0] if unit.coordinates else np.nan for unit in units_list],
                dtype=np.float32
            )
            longitudes = np.array(
                [unit.coordinates[1] if unit.coordinates else np.nan for unit in units_list],
                dtype=np.float32
            )

            self._create_dataset(geo_group, 'latitudes', latitudes)
            self._create_dataset(geo_group, 'longitudes', longitudes)

        logger.info(f"  Wrote {num_units} geographical units (core)")

    def _write_geography_properties(self, f, world):
        """Write configured geography property datasets to HDF5."""
        units_list = self._geo_units_sorted

        # Flushed cohort must not have drifted: columns written at flush
        # time are aligned to the cohort order recorded then.
        current_cohort_ids = [unit.id for unit in units_list]
        self._geography_flush.check_drift(current_cohort_ids)

        geo_settings = self.config.get_geography_settings()
        # Columns already written by flush_properties steps are skipped:
        # their in-memory values are tombstoned (may/flushing.py).
        properties_to_include = [
            name for name in geo_settings['properties']
            if name not in self._geography_flush.flushed_names
        ]
        if not properties_to_include:
            return

        geo_group = f['geography']
        props_group = geo_group.require_group('properties')
        for prop_name in properties_to_include:
            self._write_property_array(props_group, prop_name, units_list)
        logger.info(f"  Wrote geography properties: {properties_to_include}")

    def _stable_sort_by_geo_unit(self, objects, geo_unit_key=lambda obj: obj.geographical_unit):
        """
        Frozen row order shared by every geo-sorted cohort (population,
        geo_units, venues, subsets, and the venue-type flush cohort):
        stable sort by geo_unit id, missing units last (-1). Deterministic
        over the run (ids/geo assignment never mutate), so early core
        writes, flush steps and the final export all agree on row
        positions.

        `geo_unit_key` extracts the GeographicalUnit to sort by (or None);
        defaults to `obj.geographical_unit` for objects that carry it
        directly (people, venues). Subsets pass
        `lambda subset: subset.venue.geographical_unit`; geo_units pass
        `lambda unit: unit` to sort by their own id.
        """
        geo_unit_ids_raw = np.array(
            [(geo_unit_key(obj).id if geo_unit_key(obj) else -1) for obj in objects],
            dtype=np.int32
        )
        sort_idx = np.argsort(geo_unit_ids_raw, kind='stable')
        objects_sorted = [objects[i] for i in sort_idx]
        return objects_sorted, geo_unit_ids_raw[sort_idx]

    def _cached_or_resorted(self, cached_sorted, live_count, compute_fresh):
        """
        Reuse `cached_sorted` (a previous flush call's sorted cohort) when
        its length still matches `live_count`, else call `compute_fresh()`
        for a full resort. See docs/adr/0006: a count match is trusted as
        sufficient evidence the cohort hasn't drifted, without re-verifying
        order — CohortFlushTracker.record_flush()/check_drift() remain the
        backstop that catches a genuine drift (count mismatch, or anything
        export() re-derives fresh and compares against the recorded
        baseline).
        """
        if cached_sorted is not None and len(cached_sorted) == live_count:
            return cached_sorted
        return compute_fresh()

    def _write_population_core(self, f, world):
        """Write population core datasets (no properties) to HDF5."""
        pop_group = f.create_group('population')

        people_sorted, geo_unit_ids_sorted = self._stable_sort_by_geo_unit(world.population.people)

        # Store sorted people for property serialization within this
        # export() call (mirrors self._geo_units_sorted for geography).
        self._people_sorted = people_sorted

        people = world.population.people
        if not people:
            logger.warning("No people to serialize")
            return

        num_people = len(people)

        logger.info(f"  Serializing {num_people:,} people...")
        logger.info(f"    ✓ Sorted {num_people:,} people by geo_unit_id")

        logger.info(f"    Building partition index...")
        self._write_partition_index(pop_group, geo_unit_ids_sorted)
        logger.info(f"    ✓ Wrote partition index")

        logger.info(f"    Writing core attributes in chunks...")
        
        chunk_size = 100000
        sex_map = {"male": 0, "female": 1, "": 2, "unknown": 2}
        
        # Initialize datasets
        ids_ds = self._create_empty_dataset(pop_group, 'ids', np.int32, (num_people,))
        ages_ds = self._create_empty_dataset(pop_group, 'ages', np.float32, (num_people,))
        sexes_ds = self._create_empty_dataset(pop_group, 'sexes', np.uint8, (num_people,))
        geo_ds = self._create_empty_dataset(pop_group, 'geo_unit_ids', np.int32, (num_people,))

        for i in range(0, num_people, chunk_size):
            end = min(i + chunk_size, num_people)
            chunk = people_sorted[i:end]
            
            ids_chunk = np.array([p.id for p in chunk], dtype=np.int32)
            ages_chunk = np.array([p.age for p in chunk], dtype=np.float32)
            sexes_chunk = np.array([sex_map.get(p.sex.lower(), 2) for p in chunk], dtype=np.uint8)
            geo_chunk = geo_unit_ids_sorted[i:end]
            
            ids_ds[i:end] = ids_chunk
            ages_ds[i:end] = ages_chunk
            sexes_ds[i:end] = sexes_chunk
            geo_ds[i:end] = geo_chunk
            
            if (i // chunk_size) % 5 == 0:
                logger.info(f"      Processed {end:,}/{num_people:,} people...")

        logger.info(f"    ✓ Wrote core datasets to HDF5")
        logger.info(f"  Wrote {num_people:,} people (core)")

    def _write_population_properties(self, f, world):
        """Write configured person property datasets to HDF5."""
        # Flushed cohort must not have drifted: columns written at flush
        # time are aligned to the cohort order recorded then.
        current_cohort_ids = [person.id for person in (self._people_sorted or [])]
        self._population_flush.check_drift(current_cohort_ids)

        # Columns already written by flush_properties steps are skipped:
        # their in-memory values are tombstoned (may/flushing.py).
        properties_to_include = [
            name for name in self.config.get_person_properties()
            if name not in self._population_flush.flushed_names
        ]
        if not properties_to_include:
            return

        pop_group = f['population']
        props_group = pop_group.require_group('properties')

        for prop_idx, prop_name in enumerate(properties_to_include, 1):
            logger.info(f"    Writing property {prop_idx}/{len(properties_to_include)}: {prop_name}...")
            self._write_property_array(props_group, prop_name, self._people_sorted)

        logger.info(f"    Including properties: {properties_to_include}")

    def _write_partition_index(self, pop_group, geo_unit_ids):
        """
        Write partition index for efficient geo_unit-based loading.

        Creates index structure that maps geo_unit_id -> (start_index, count)
        allowing efficient range-based reads for partitioned loading.

        Args:
            pop_group: HDF5 population group
            geo_unit_ids: Sorted array of geo_unit_ids for all people

        Structure created:
            /population/partition_index/
                geo_unit_ids: [1, 2, 3, ...] - unique geo_unit IDs
                start_indices: [0, 100000, 250000, ...] - start row for each geo_unit
                counts: [100000, 150000, 50000, ...] - number of people per geo_unit
        """
        index_group = pop_group.create_group('partition_index')

        # Find unique geo_unit_ids and their boundaries
        unique_geo_units = []
        start_indices = []
        counts = []

        if len(geo_unit_ids) == 0:
            # Empty population
            logger.warning("Empty population - no partition index to create")
            return

        current_geo_unit = geo_unit_ids[0]
        current_start = 0
        current_count = 0

        for i, geo_unit_id in enumerate(geo_unit_ids):
            if geo_unit_id != current_geo_unit:
                # Save previous geo_unit
                unique_geo_units.append(current_geo_unit)
                start_indices.append(current_start)
                counts.append(current_count)

                # Start new geo_unit
                current_geo_unit = geo_unit_id
                current_start = i
                current_count = 1
            else:
                current_count += 1

        # Save last geo_unit
        unique_geo_units.append(current_geo_unit)
        start_indices.append(current_start)
        counts.append(current_count)

        # Convert to numpy arrays
        unique_geo_units = np.array(unique_geo_units, dtype=np.int32)
        start_indices = np.array(start_indices, dtype=np.int32)
        counts = np.array(counts, dtype=np.int32)

        # Write datasets
        self._create_dataset(index_group, 'geo_unit_ids', unique_geo_units)
        self._create_dataset(index_group, 'start_indices', start_indices)
        self._create_dataset(index_group, 'counts', counts)

        logger.info(f"      Created partition index for {len(unique_geo_units)} geo_units")
        logger.info(f"      Min people per geo_unit: {counts.min()}")
        logger.info(f"      Max people per geo_unit: {counts.max()}")
        logger.info(f"      Avg people per geo_unit: {counts.mean():.1f}")

    def _write_activity_mapping_partition_index(self, activity_map_group, people_sorted, activity_offsets, total_activity_mappings):
        """
        Write partition index for efficient geo_unit-based activity mapping loading.

        Creates index structure that maps geo_unit_id -> (start_row, count)
        for the activity_data array, allowing efficient range-based reads.

        Args:
            activity_map_group: HDF5 activity_map group
            people_sorted: People list sorted by geo_unit_id
            activity_offsets: Array of start indices for each person's activity mappings
            total_activity_mappings: Total number of rows in activity_data

        Structure created:
            /activity_mappings/activity_map/partition_index/
                geo_unit_ids: [1, 2, 3, ...] - unique geo_unit IDs
                start_indices: [0, 500000, 1250000, ...] - start row in activity_data
                counts: [500000, 750000, 300000, ...] - number of mapping rows per geo_unit
        """
        index_group = activity_map_group.create_group('partition_index')

        if len(people_sorted) == 0:
            logger.warning("Empty population - no activity mapping partition index to create")
            return

        # Group people by geo_unit and track activity mapping row ranges
        unique_geo_units = []
        start_indices = []
        counts = []

        current_geo_unit = people_sorted[0].geographical_unit.id if people_sorted[0].geographical_unit else -1
        current_start_row = 0  # Start row in activity_data for this geo_unit

        for person_idx, person in enumerate(people_sorted):
            geo_unit_id = person.geographical_unit.id if person.geographical_unit else -1

            if geo_unit_id != current_geo_unit:
                # Save previous geo_unit's activity mapping range
                # End row is the start of current person's activity mappings
                end_row = activity_offsets[person_idx] if person_idx < len(activity_offsets) else total_activity_mappings
                activity_mappings_count = end_row - current_start_row

                unique_geo_units.append(current_geo_unit)
                start_indices.append(current_start_row)
                counts.append(activity_mappings_count)

                # Start new geo_unit
                current_geo_unit = geo_unit_id
                current_start_row = end_row

        # Save last geo_unit (activity mappings extend to end of activity_data)
        activity_mappings_count = total_activity_mappings - current_start_row
        unique_geo_units.append(current_geo_unit)
        start_indices.append(current_start_row)
        counts.append(activity_mappings_count)

        # Convert to numpy arrays
        unique_geo_units = np.array(unique_geo_units, dtype=np.int32)
        start_indices = np.array(start_indices, dtype=np.int32)
        counts = np.array(counts, dtype=np.int32)

        # Write datasets
        self._create_dataset(index_group, 'geo_unit_ids', unique_geo_units)
        self._create_dataset(index_group, 'start_indices', start_indices)
        self._create_dataset(index_group, 'counts', counts)

        logger.info(f"      Created activity mapping partition index for {len(unique_geo_units)} geo_units")
        if len(counts) > 0:
            logger.info(f"      Min mappings per geo_unit: {counts.min()}")
            logger.info(f"      Max mappings per geo_unit: {counts.max()}")
            logger.info(f"      Avg mappings per geo_unit: {counts.mean():.1f}")

    def _write_subset_metadata_partition_index(self, subsets_group, all_subsets_sorted):
        """
        Write partition index for efficient geo_unit-based subset metadata loading.

        Creates index structure that maps geo_unit_id -> (start_index, count)
        for the subset metadata arrays (venue_ids, subset_indices, etc.),
        allowing efficient range-based reads without scanning all 35M venue_ids.

        Args:
            subsets_group: HDF5 subsets group
            all_subsets_sorted: Subsets list sorted by venue's geo_unit_id

        Structure created:
            /venues/subsets/partition_index/
                geo_unit_ids: [1, 2, 3, ...] - unique geo_unit IDs
                start_indices: [0, 1000, 3500, ...] - start row in subset arrays
                counts: [1000, 2500, 750, ...] - number of subsets per geo_unit
        """
        index_group = subsets_group.create_group('partition_index')

        if len(all_subsets_sorted) == 0:
            logger.warning("Empty subsets - no metadata partition index to create")
            return

        # Find unique geo_unit_ids and their boundaries
        unique_geo_units = []
        start_indices = []
        counts = []

        current_geo_unit = all_subsets_sorted[0].venue.geographical_unit.id if all_subsets_sorted[0].venue.geographical_unit else -1
        current_start = 0
        current_count = 0

        for i, subset in enumerate(all_subsets_sorted):
            geo_unit_id = subset.venue.geographical_unit.id if subset.venue.geographical_unit else -1

            if geo_unit_id != current_geo_unit:
                # Save previous geo_unit
                unique_geo_units.append(current_geo_unit)
                start_indices.append(current_start)
                counts.append(current_count)

                # Start new geo_unit
                current_geo_unit = geo_unit_id
                current_start = i
                current_count = 1
            else:
                current_count += 1

        # Save last geo_unit
        unique_geo_units.append(current_geo_unit)
        start_indices.append(current_start)
        counts.append(current_count)

        # Convert to numpy arrays
        unique_geo_units = np.array(unique_geo_units, dtype=np.int32)
        start_indices = np.array(start_indices, dtype=np.int32)
        counts = np.array(counts, dtype=np.int32)

        # Write datasets
        self._create_dataset(index_group, 'geo_unit_ids', unique_geo_units)
        self._create_dataset(index_group, 'start_indices', start_indices)
        self._create_dataset(index_group, 'counts', counts)

        logger.info(f"      Created subset metadata partition index for {len(unique_geo_units)} geo_units")
        if len(counts) > 0:
            logger.info(f"      Min subsets per geo_unit: {counts.min()}")
            logger.info(f"      Max subsets per geo_unit: {counts.max()}")
            logger.info(f"      Avg subsets per geo_unit: {counts.mean():.1f}")

    def _write_subset_members_partition_index(self, subsets_group, all_subsets_sorted, members_offsets, total_members):
        """
        Write partition index for efficient geo_unit-based subset membership loading.

        Creates index structure that maps geo_unit_id -> (start_row, count)
        for the members_flat array, allowing efficient range-based reads.

        Args:
            subsets_group: HDF5 subsets group
            all_subsets_sorted: Subsets list sorted by venue's geo_unit_id
            members_offsets: Array of start indices for each subset's members in members_flat
            total_members: Total number of entries in members_flat

        Structure created:
            /venues/subsets/members_partition_index/
                geo_unit_ids: [1, 2, 3, ...] - unique geo_unit IDs
                start_indices: [0, 50000, 125000, ...] - start row in members_flat
                counts: [50000, 75000, 30000, ...] - number of members per geo_unit
        """
        index_group = subsets_group.create_group('members_partition_index')

        if len(all_subsets_sorted) == 0:
            logger.warning("Empty subsets - no partition index to create")
            return

        # Group subsets by geo_unit and track membership row ranges
        unique_geo_units = []
        start_indices = []
        counts = []

        current_geo_unit = all_subsets_sorted[0].venue.geographical_unit.id if all_subsets_sorted[0].venue.geographical_unit else -1
        current_start_row = 0  # Start row in members_flat for this geo_unit

        for subset_idx, subset in enumerate(all_subsets_sorted):
            geo_unit_id = subset.venue.geographical_unit.id if subset.venue.geographical_unit else -1

            if geo_unit_id != current_geo_unit:
                # Save previous geo_unit's membership range
                # End row is the start of current subset's members
                end_row = members_offsets[subset_idx] if subset_idx < len(members_offsets) else total_members
                member_count = end_row - current_start_row

                unique_geo_units.append(current_geo_unit)
                start_indices.append(current_start_row)
                counts.append(member_count)

                # Start new geo_unit
                current_geo_unit = geo_unit_id
                current_start_row = end_row

        # Save last geo_unit (members extend to end of members_flat)
        member_count = total_members - current_start_row
        unique_geo_units.append(current_geo_unit)
        start_indices.append(current_start_row)
        counts.append(member_count)

        # Convert to numpy arrays
        unique_geo_units = np.array(unique_geo_units, dtype=np.int32)
        start_indices = np.array(start_indices, dtype=np.int32)
        counts = np.array(counts, dtype=np.int32)

        # Write datasets
        self._create_dataset(index_group, 'geo_unit_ids', unique_geo_units)
        self._create_dataset(index_group, 'start_indices', start_indices)
        self._create_dataset(index_group, 'counts', counts)

        logger.info(f"      Created subset partition index for {len(unique_geo_units)} geo_units")
        if len(counts) > 0:
            logger.info(f"      Min members per geo_unit: {counts.min()}")
            logger.info(f"      Max members per geo_unit: {counts.max()}")
            logger.info(f"      Avg members per geo_unit: {counts.mean():.1f}")

    def _write_venue_partition_index(self, venues_group, all_venues_sorted):
        """
        Write partition index for efficient geo_unit-based venue loading.

        Creates index structure that maps geo_unit_id -> (start_index, count)
        for the venue arrays, allowing efficient range-based reads.

        Args:
            venues_group: HDF5 venues group
            all_venues_sorted: Venues list sorted by geo_unit_id

        Structure created:
            /venues/partition_index/
                geo_unit_ids: [1, 2, 3, ...] - unique geo_unit IDs
                start_indices: [0, 100, 350, ...] - start row in venue arrays
                counts: [100, 250, 75, ...] - number of venues per geo_unit
        """
        index_group = venues_group.create_group('partition_index')

        if len(all_venues_sorted) == 0:
            logger.warning("Empty venues - no partition index to create")
            return

        # Find unique geo_unit_ids and their boundaries
        unique_geo_units = []
        start_indices = []
        counts = []

        current_geo_unit = all_venues_sorted[0].geographical_unit.id if all_venues_sorted[0].geographical_unit else -1
        current_start = 0
        current_count = 0

        for i, venue in enumerate(all_venues_sorted):
            geo_unit_id = venue.geographical_unit.id if venue.geographical_unit else -1

            if geo_unit_id != current_geo_unit:
                # Save previous geo_unit
                unique_geo_units.append(current_geo_unit)
                start_indices.append(current_start)
                counts.append(current_count)

                # Start new geo_unit
                current_geo_unit = geo_unit_id
                current_start = i
                current_count = 1
            else:
                current_count += 1

        # Save last geo_unit
        unique_geo_units.append(current_geo_unit)
        start_indices.append(current_start)
        counts.append(current_count)

        # Convert to numpy arrays
        unique_geo_units = np.array(unique_geo_units, dtype=np.int32)
        start_indices = np.array(start_indices, dtype=np.int32)
        counts = np.array(counts, dtype=np.int32)

        # Write datasets
        self._create_dataset(index_group, 'geo_unit_ids', unique_geo_units)
        self._create_dataset(index_group, 'start_indices', start_indices)
        self._create_dataset(index_group, 'counts', counts)

        logger.info(f"      Created venue partition index for {len(unique_geo_units)} geo_units")
        logger.info(f"      Min venues per geo_unit: {counts.min()}")
        logger.info(f"      Max venues per geo_unit: {counts.max()}")
        logger.info(f"      Avg venues per geo_unit: {counts.mean():.1f}")

    def _write_venues(self, f, world):
        """Write venues and subsets to HDF5."""
        # May already exist: flush_venue_properties() creates /venues/properties
        # mid-timeline in the canonical file.
        venues_group = f.require_group('venues')
        venue_global_settings = self.config.get_venue_global_settings()

        # Get all venues as a list. No early-return guard for the empty
        # case: _write_venue_properties' drift check (docs/adr/0005) must
        # still run when every venue has been removed post-flush, and the
        # downstream per-step writers already handle a zero-length list.
        all_venues = world.venues.get_all_venues_list()
        num_venues = len(all_venues)

        # Sort venues by their geographical unit ID
        all_venues_sorted, _ = self._stable_sort_by_geo_unit(all_venues)

        logger.info(f"    ✓ Sorted {num_venues:,} venues by geo_unit_id")

        # Assign global IDs (0, 1, 2, ..., N-1) to SORTED venues
        global_ids = np.arange(num_venues, dtype=np.int32)

        # Create mapping for faster lookup during activity map export
        self._venue_to_global_id = {}
        for v, global_id in zip(all_venues_sorted, global_ids):
            self._venue_to_global_id[id(v)] = global_id

        # Compute ranks_in_type: sequential position within each type in sorted order.
        # Must match the indexing in _write_venue_properties.
        type_counters = {}
        ranks_in_type_list = []
        for v in all_venues_sorted:
            rank = type_counters.get(v.type, 0)
            ranks_in_type_list.append(rank)
            type_counters[v.type] = rank + 1
        type_scoped_ids = np.array(ranks_in_type_list, dtype=np.int32)

        # Core attributes (always included)
        ids = global_ids  # Use GLOBAL IDs for C++
        
        # Move names to metadata to save memory in core simulation loop
        names = np.array([v.name for v in all_venues_sorted], dtype=h5py.string_dtype())
        metadata_group = f.require_group('metadata/names')
        self._create_dataset(metadata_group, 'venues', names)

        # Convert venue types to uint8 Enum
        unique_types = sorted(list(set(v.type for v in all_venues_sorted)))
        type_to_id = {t: i for i, t in enumerate(unique_types)}
        types = np.array([type_to_id[v.type] for v in all_venues_sorted], dtype=np.uint8)

        # Store the type mapping for C++ consumption
        self._venue_type_id_map = type_to_id

        # Geographical unit IDs (where venue is located)
        geo_unit_ids = np.array(
            [v.geographical_unit.id if v.geographical_unit else -1 for v in all_venues_sorted],
            dtype=np.int32
        )

        # Parent venue IDs (-1 for root venues)
        # IMPORTANT: Use global IDs for parents too!
        parent_ids = np.array(
            [self._venue_to_global_id.get(id(v.parent), -1) if v.parent else -1 for v in all_venues_sorted],
            dtype=np.int32
        )

        # Write core datasets
        self._create_dataset(venues_group, 'ids', ids)
        self._create_dataset(venues_group, 'types', types)
        self._create_dataset(venues_group, 'ranks_in_type', type_scoped_ids) # Needed for lazy property loading
        self._create_dataset(venues_group, 'geo_unit_ids', geo_unit_ids)
        self._create_dataset(venues_group, 'parent_ids', parent_ids)

        # Coordinates (optional)
        if venue_global_settings.get('include_coordinates', True):
            latitudes = np.array(
                [v.coordinates[0] if v.coordinates else np.nan for v in all_venues_sorted],
                dtype=np.float32
            )
            longitudes = np.array(
                [v.coordinates[1] if v.coordinates else np.nan for v in all_venues_sorted],
                dtype=np.float32
            )

            self._create_dataset(venues_group, 'latitudes', latitudes)
            self._create_dataset(venues_group, 'longitudes', longitudes)

        # is_residence flag (optional)
        if venue_global_settings.get('include_is_residence', True):
            is_residence = np.array(
                [v.properties.get('is_residence', False) for v in all_venues_sorted],
                dtype=np.bool_
            )
            self._create_dataset(venues_group, 'is_residence', is_residence)

        logger.info(f"    Building venue partition index...")
        self._write_venue_partition_index(venues_group, all_venues_sorted)
        logger.info(f"    ✓ Wrote venue partition index")

        # Properties (per-type configuration)
        self._write_venue_properties(venues_group, all_venues_sorted)

        # Write subsets
        num_subsets = self._write_subsets(venues_group, all_venues_sorted)

        logger.info(f"  Wrote {num_venues:,} venues")
        logger.info(f"  Wrote {num_subsets:,} subsets")

        return num_subsets

    def _write_venue_properties(self, venues_group, all_venues):
        """Write venue properties based on per-type configuration."""
        # Group venues by type
        venues_by_type = defaultdict(list)
        for v in all_venues:
            venues_by_type[v.type].append(v)

        # For each type, write configured properties
        props_group = venues_group.require_group('properties')

        # Flushed cohorts must not have drifted: columns written at flush
        # time are aligned to the cohort order recorded then. Checked over
        # every tracked type up front (self._venue_flushes), independent of
        # venues_by_type, so a type with zero current venues (fully removed
        # post-flush) is still caught rather than silently skipped — see
        # docs/adr/0005.
        for venue_type, tracker in self._venue_flushes.items():
            current_cohort_ids = [venue.id for venue in venues_by_type.get(venue_type, [])]
            tracker.check_drift(current_cohort_ids)

        # Subset-flushed cohorts must be fully closed: no live (non-gutted)
        # subset may exist on the venue_type, and every staged subset must
        # still be present as a husk — see docs/adr/0005, flush_venue_subsets.
        from may.flushing import GuttedMembers
        for venue_type, tracker in self._venue_flushes.items():
            venues_of_type = venues_by_type.get(venue_type, [])
            live_keys = {
                (venue.id, subset.subset_index)
                for venue in venues_of_type
                for subset in venue.subsets.values()
                if not isinstance(subset.members, GuttedMembers)
            }
            all_keys = {
                (venue.id, subset.subset_index)
                for venue in venues_of_type
                for subset in venue.subsets.values()
            }
            tracker.check_subset_drift(live_keys, all_keys)

        # Configured types with zero current venues (e.g. never populated,
        # or fully removed post-flush) must still get their subgroup —
        # mirrors the drift-check loop above, not just venues_by_type.
        all_venue_types = set(venues_by_type) | set(self.config.venue_type_properties)
        for venue_type in all_venue_types:
            venues = venues_by_type.get(venue_type, [])
            # Columns already written by flush_properties steps are skipped:
            # their in-memory values are tombstoned (may/flushing.py).
            tracker = self._venue_flushes.get(venue_type)
            flushed = tracker.flushed_names if tracker else set()
            properties_to_include = [
                name for name in self.config.get_venue_properties(venue_type)
                if name not in flushed
            ]

            if not properties_to_include:
                continue

            type_group = props_group.require_group(venue_type)

            for prop_name in properties_to_include:
                # Create array for this property across all venues of this type
                self._write_property_array(type_group, prop_name, venues)

            logger.info(f"    {venue_type}: {len(properties_to_include)} properties ({len(venues)} venues)")

    def _write_subsets(self, venues_group, all_venues):
        """Write subset data to HDF5."""
        subsets_group = venues_group.create_group('subsets')

        # Flatten all subsets from all venues
        all_subsets = []
        for venue in all_venues:
            for subset in venue.subsets.values():
                all_subsets.append(subset)

        if not all_subsets:
            logger.warning("No subsets to serialize")
            return 0

        num_subsets = len(all_subsets)

        # Sort subsets by their venue's geographical unit ID
        all_subsets_sorted, _ = self._stable_sort_by_geo_unit(
            all_subsets, geo_unit_key=lambda subset: subset.venue.geographical_unit
        )

        logger.info(f"    ✓ Sorted {num_subsets:,} subsets by venue's geo_unit_id")

        # Core attributes
        # IMPORTANT: Use global venue IDs
        venue_ids = np.array([self._venue_to_global_id[id(s.venue)] for s in all_subsets_sorted], dtype=np.int32)
        subset_indices = np.array([s.subset_index for s in all_subsets_sorted], dtype=np.int32)
        # Move subset names to metadata
        subset_names = np.array([s.subset_name for s in all_subsets_sorted], dtype=h5py.string_dtype())
        metadata_group = venues_group.file.require_group('metadata/names')
        self._create_dataset(metadata_group, 'subsets', subset_names)

        # Populate subset_names registry for parallel consistency
        unique_subset_names = sorted(list(set(s.subset_name for s in all_subsets_sorted)))
        self.registries['subset_names'] = {name: i for i, name in enumerate(unique_subset_names)}

        # Member counts (useful for C++)
        member_counts = np.array([len(s.members) for s in all_subsets_sorted], dtype=np.int32)

        logger.info(f"    Building subset metadata partition index...")
        self._write_subset_metadata_partition_index(subsets_group, all_subsets_sorted)
        logger.info(f"    ✓ Wrote subset metadata partition index")

        # Write datasets
        self._create_dataset(subsets_group, 'venue_ids', venue_ids)
        self._create_dataset(subsets_group, 'subset_indices', subset_indices)
        self._create_dataset(subsets_group, 'member_counts', member_counts)

        # Write member lists (ragged array - need special handling)
        self._write_subset_members(subsets_group, all_subsets_sorted)

        return num_subsets

    def _write_subset_members(self, subsets_group, all_subsets):
        """
        Write subset member lists as ragged arrays with chunked processing.
        Gutted subsets (flushed to staging, see flush_venue_subsets) have
        their member ids streamed back in from the staging reader instead
        of iterating the live (sentinel) `.members`.
        """
        from may.flushing import GuttedMembers

        num_subsets = len(all_subsets)
        chunk_size = 200000

        # Pass 1: Count total members
        logger.info(f"    Counting total subset members...")
        total_members = sum(len(s.members) for s in all_subsets)
        logger.info(f"    Total subset memberships to write: {total_members:,}")

        # Initialize datasets
        members_ds = self._create_empty_dataset(subsets_group, 'members_flat', np.int32, (total_members,))
        offsets_ds = self._create_empty_dataset(subsets_group, 'members_offsets', np.int32, (num_subsets,))

        current_member_idx = 0
        all_offsets = []

        logger.info(f"    Writing subset memberships in chunks...")
        for i in range(0, num_subsets, chunk_size):
            end = min(i + chunk_size, num_subsets)
            chunk = all_subsets[i:end]
            
            chunk_members = []
            chunk_offsets = []
            
            for subset in chunk:
                chunk_offsets.append(current_member_idx)
                if isinstance(subset.members, GuttedMembers):
                    ids = self._staged_member_ids(subset)
                else:
                    ids = [p.id for p in subset.members]
                chunk_members.extend(ids)
                current_member_idx += len(ids)
            
            # Write chunk to HDF5
            if chunk_members:
                members_ds[chunk_offsets[0]:current_member_idx] = np.array(chunk_members, dtype=np.int32)
            
            offsets_ds[i:end] = np.array(chunk_offsets, dtype=np.int32)
            
            if (i // chunk_size) % 5 == 0:
                logger.info(f"      Processed {end:,}/{num_subsets:,} subsets...")

        logger.info(f"    Building subset members partition index...")
        # Get offsets back for partition index
        offsets_full = offsets_ds[:]
        self._write_subset_members_partition_index(subsets_group, all_subsets, offsets_full, total_members)
        logger.info(f"    ✓ Wrote subset members partition index")

        logger.info(f"    Total subset memberships: {total_members:,}")

    def _write_activity_mappings(self, f, world):
        """Write activity mapping data (activity_map, hierarchies)."""
        rel_group = f.create_group('activity_mappings')

        # Activity map (person → venues via activities)
        if self.config.should_include_activity_map():
            # Use sorted people order (same as population)
            people_sorted = getattr(self, '_people_sorted', world.population.people)
            self._write_activity_map(rel_group, world, people_sorted)

        # Generic per-membership numeric metadata side-table. Rows only for
        # subsets whose Subset.member_metadata dict is populated (e.g.
        # transport-line legs carrying (t_board_min, t_alight_min)).
        self._write_membership_metadata(rel_group, world)

    def _write_activity_map(self, rel_group, world, people_sorted):
        """
        Write activity_map data with chunked processing for memory efficiency.
        """
        if not people_sorted:
            logger.warning("No people to serialize activity map for")
            return

        activity_map_group = rel_group.create_group('activity_map')


        # Collect activity names
        activity_names_set = set()
        for person in people_sorted:
            activity_names_set.update(person.activities)

        activity_names = sorted(list(activity_names_set))
        activity_to_idx = {name: idx for idx, name in enumerate(activity_names)}

        # Write activity names
        activity_names_array = np.array(activity_names, dtype=h5py.string_dtype())
        self._create_dataset(activity_map_group, 'activity_names', activity_names_array)

        # Prepare for chunked processing
        num_people = len(people_sorted)
        chunk_size = 200000
        venue_to_id = self._venue_to_global_id
        
        # Two passes: 1. Count total 2. Write.
        
        logger.info(f"    Counting activity mappings...")
        mapping_counts = []
        total_mappings = 0
        for person in people_sorted:
            count = 0
            for name, types in person.activity_map.items():
                if name in activity_to_idx:
                    for subsets_list in types.values():
                        count += len(subsets_list)
            mapping_counts.append(count)
            total_mappings += count
        
        logger.info(f"    Total activity mappings to write: {total_mappings:,}")
        
        # Initialize datasets
        activity_ds = self._create_empty_dataset(activity_map_group, 'activity_data', np.int32, (total_mappings, 4))
        offsets_ds = self._create_empty_dataset(activity_map_group, 'activity_offsets', np.int32, (num_people,))
        
        current_mapping_idx = 0
        activity_offsets = []

        logger.info(f"    Writing activity mappings in chunks...")
        for i in range(0, num_people, chunk_size):
            end = min(i + chunk_size, num_people)
            chunk = people_sorted[i:end]
            
            chunk_p_ids = []
            chunk_a_idxs = []
            chunk_v_ids = []
            chunk_s_idxs = []
            chunk_offsets = []
            
            for person in chunk:
                chunk_offsets.append(current_mapping_idx)
                person_id = person.id
                for name, types in person.activity_map.items():
                    if name in activity_to_idx:
                        act_idx = activity_to_idx[name]
                        for subsets_list in types.values():
                            for subset in subsets_list:
                                v_id = id(subset.venue)
                                if v_id in venue_to_id:
                                    chunk_p_ids.append(person_id)
                                    chunk_a_idxs.append(act_idx)
                                    chunk_v_ids.append(venue_to_id[v_id])
                                    chunk_s_idxs.append(subset.subset_index)
                                    current_mapping_idx += 1
            
            # Write chunk to HDF5
            if chunk_p_ids:
                chunk_data = np.empty((len(chunk_p_ids), 4), dtype=np.int32)
                chunk_data[:, 0] = chunk_p_ids
                chunk_data[:, 1] = chunk_a_idxs
                chunk_data[:, 2] = chunk_v_ids
                chunk_data[:, 3] = chunk_s_idxs
                
                start_row = chunk_offsets[0]
                activity_ds[start_row:current_mapping_idx] = chunk_data
            
            offsets_ds[i:end] = np.array(chunk_offsets, dtype=np.int32)
            
            if (i // chunk_size) % 5 == 0:
                logger.info(f"      Processed {end:,}/{num_people:,} people...")

        logger.info(f"  Building activity mapping partition index...")

        offsets_full = offsets_ds[:] 
        self._write_activity_mapping_partition_index(activity_map_group, people_sorted, offsets_full, total_mappings)
        logger.info(f"    ✓ Wrote activity mapping partition index")

        logger.info(f"  Activity map: {len(activity_names)} unique activities:")
        for name in activity_names:
            logger.info(f"                                           - {name}")
        logger.info(f"    Total activity mappings: {total_mappings:,}")

    def _write_membership_metadata(self, rel_group, world):
        """Write Subset.member_metadata as a generic side-table.

        Produces /activity_mappings/membership_metadata with one row per
        (person, venue) membership that carries metadata:
            person_id    : int32
            venue_id     : int32 (global venue id, matches activity_data col 2)
            <field_name> : float32 — one column per metadata field name found

        Field names are discovered by scanning all subsets' member_metadata
        dicts. Datasets are float32 to accept both ints and floats; -1.0 is
        the sentinel for "field absent for this row" (rare; should not occur
        unless different memberships in the same export carry disjoint
        metadata schemas).

        Two passes over the venues fill preallocated numpy arrays (4 bytes
        per row per column), so memory stays flat at 60M-scale ridership. Any
        value above 2**24 is logged loudly: float32 represents integers
        exactly only up to that point, so e.g. venue ids in big worlds could
        corrupt.
        """
        from may.flushing import GuttedMembers

        venue_to_global_id = self._venue_to_global_id

        # Pass 1: count rows and discover the union of field names (in
        # discovery order, so output column order is deterministic).
        n_rows = 0
        field_names = []
        field_set = set()
        for venue in world.venues.get_all_venues_list():
            if id(venue) not in venue_to_global_id:
                continue
            for subset in venue.subsets.values():
                if isinstance(subset.members, GuttedMembers):
                    # Gutted subsets contribute zero metadata rows in Phase
                    # B (households never populate member_metadata); real
                    # per-subset retrieval is a Phase C concern.
                    continue
                meta = getattr(subset, 'member_metadata', None)
                if not meta:
                    continue
                n_rows += len(meta)
                for fields in meta.values():
                    for fname in fields.keys():
                        if fname not in field_set:
                            field_set.add(fname)
                            field_names.append(fname)

        if n_rows == 0:
            logger.info("  No membership_metadata to serialise (no subsets populated)")
            return

        # Pass 2: fill. Same venue/subset/dict iteration order as pass 1
        # (nothing mutates between passes), so row k lands in slot k.
        person_arr = np.empty(n_rows, dtype=np.int32)
        venue_arr = np.empty(n_rows, dtype=np.int32)
        per_field = {f: np.full(n_rows, -1.0, dtype=np.float32) for f in field_names}
        k = 0
        for venue in world.venues.get_all_venues_list():
            global_id = venue_to_global_id.get(id(venue))
            if global_id is None:
                continue
            for subset in venue.subsets.values():
                if isinstance(subset.members, GuttedMembers):
                    continue
                meta = getattr(subset, 'member_metadata', None)
                if not meta:
                    continue
                for pid, fields in meta.items():
                    person_arr[k] = pid
                    venue_arr[k] = global_id
                    for fname, val in fields.items():
                        per_field[fname][k] = float(val)
                    k += 1

        # float32 keeps integers exact only up to 2**24; beyond that ids
        # round silently. Geo unit ids are safely small; this trips if a
        # config ever routes venue-scale ids through the side-table.
        for fname, arr in per_field.items():
            if np.abs(arr).max() > 2 ** 24:
                logger.warning(
                    f"  membership_metadata field '{fname}' holds values above "
                    f"2^24 — float32 storage rounds such integers; ids stored "
                    f"in this field may be corrupt"
                )

        meta_group = rel_group.create_group('membership_metadata')
        self._create_dataset(meta_group, 'person_ids', person_arr)
        self._create_dataset(meta_group, 'venue_ids', venue_arr)
        # Field-name registry (string array, ordered) so consumers can iterate.
        self._create_dataset(
            meta_group, 'field_names',
            np.array(field_names, dtype=h5py.string_dtype()),
        )
        for fname in field_names:
            self._create_dataset(meta_group, fname, per_field[fname])
        logger.info(
            f"  Wrote membership_metadata side-table: {n_rows:,} rows × "
            f"{len(field_names)} fields {field_names}"
        )

    def _write_property_array(self, group, prop_name, objects):
        """
        Write a property array for a list of objects in chunks.
        Supports different property types (int, float, str, bool, list, dict).
        """
        num_objects = len(objects)
        chunk_size = 100000
        
        # Step 1: Infer type from first non-None value
        sample_val = None
        for i in range(0, num_objects, chunk_size):
            chunk_slice = objects[i:min(i + chunk_size, num_objects)]
            for obj in chunk_slice:
                val = obj.properties.get(prop_name)
                if val is not None:
                    sample_val = val
                    break
            if sample_val is not None:
                break
        
        if sample_val is None:
            logger.debug(f"Skipping property '{prop_name}' (all None)")
            return

        # Step 2: Determine dtype and create dataset
        dtype = None
        fill_value = None
        
        if isinstance(sample_val, bool):
            dtype = np.bool_
            fill_value = False
        elif isinstance(sample_val, int):
            dtype = np.int32
            fill_value = -1
        elif isinstance(sample_val, float):
            dtype = np.float32
            fill_value = np.nan
        else:
            # Strings and complex types (JSON)
            dtype = h5py.string_dtype()
            fill_value = ""

        ds = self._create_empty_dataset(group, prop_name, dtype, (num_objects,))

        # Step 3: Write in chunks
        import json
        for i in range(0, num_objects, chunk_size):
            end = min(i + chunk_size, num_objects)
            chunk = objects[i:end]
            
            # get values once per object
            chunk_vals = []
            for obj in chunk:
                val = obj.properties.get(prop_name)
                if val is None:
                    chunk_vals.append(fill_value)
                elif isinstance(val, set):
                    chunk_vals.append(json.dumps([p.id for p in val]))
                elif isinstance(val, (list, dict)):
                    chunk_vals.append(json.dumps(val))
                else:
                    chunk_vals.append(val)
            
            ds[i:end] = np.array(chunk_vals, dtype=dtype)

    def _create_empty_dataset(self, group, name, dtype, shape):
        """Create an empty HDF5 dataset with compression."""
        compression = self.compression_settings['compression']
        compression_level = self.compression_settings['compression_level']

        # Determine HDF5 storage chunks, separate from our processing chunk_size
        # Choosing a chunk size that is a multiple of typical access patterns
        if shape and len(shape) > 0 and shape[0] > 0:
            h5_chunks = (min(shape[0], 100000),) + shape[1:]
        else:
            h5_chunks = None

        return group.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            compression=compression,
            compression_opts=compression_level,
            shuffle=True,
            chunks=h5_chunks
        )

    def _create_dataset(self, group, name, data):
        """
        Create HDF5 dataset with compression.

        Args:
            group: HDF5 group
            name: Dataset name
            data: NumPy array
        """
        compression = self.compression_settings['compression']
        compression_level = self.compression_settings['compression_level']

        # Only compress if data is large enough
        if len(data) > 100:
            # Enable shuffle=True for much better compression ratio on numeric data
            group.create_dataset(
                name,
                data=data,
                compression=compression,
                compression_opts=compression_level,
                shuffle=True
            )
        else:
            group.create_dataset(name, data=data)

    def _write_registries(self, f):
        """Write string-to-int registries for categorical data."""
        registry_group = f.create_group('metadata/registries')
        
        # Add sex mapping (predefined)
        sex_reg = registry_group.create_group('sex')
        sex_reg.attrs['mapping'] = "male:0,female:1,unknown:2"
        
        # Add venue types mapping
        if hasattr(self, '_venue_type_id_map'):
            vtype_reg = registry_group.create_dataset('venue_types', 
                                                     data=np.array(list(self._venue_type_id_map.keys()), dtype=h5py.string_dtype()))
            
        # Add geo levels mapping
        if 'geo_levels' in self.registries:
            mapping = self.registries['geo_levels']
            sorted_strings = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
            registry_group.create_dataset('geo_levels', data=np.array(sorted_strings, dtype=h5py.string_dtype()))
            
        # Add subset names mapping
        if 'subset_names' in self.registries:
            mapping = self.registries['subset_names']
            sorted_strings = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
            registry_group.create_dataset('subset_names', data=np.array(sorted_strings, dtype=h5py.string_dtype()))
            
        # Add property mappings
        props_reg_group = registry_group.create_group('properties')
        for reg_name, mapping in self.registries.items():
            if reg_name in ['geo_levels', 'subset_names']:
                continue
            prop_name = reg_name.replace("prop_", "")
            # Write unique strings in index order
            sorted_strings = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
            props_reg_group.create_dataset(prop_name, data=np.array(sorted_strings, dtype=h5py.string_dtype()))
