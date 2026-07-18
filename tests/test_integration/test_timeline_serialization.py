import pytest
import os
import h5py
from unittest.mock import patch, MagicMock

# We focus on the isolated block of logic found in run_world (Lines 335-350)
def execute_serialization_block(world_mock, config, args, logger_mock):
    """
    Simulates the exact serialization block from create_world.py
    for targeted evaluation without running the full 5-minute pipeline.
    """
    serial_config = config.get("serialization", {})
    if serial_config.get("enabled", True):
        logger_mock.info("")
        logger_mock.info("Exporting world to HDF5...")
        config_file = serial_config.get("config_file")
        if not config_file:
            raise ValueError("serialization.config_file is required")
        output_dir = serial_config.get("output_dir", ".")
        filename = serial_config.get("filename", args.filename)

        if output_dir != ".":
            os.makedirs(output_dir, exist_ok=True)

        export_path = os.path.join(output_dir, filename)
        world_mock.export_to_hdf5(export_path, config_file=config_file)


class TestCreateWorldSerializationRouting:
    """Verifies the engine parses combinations of YAML and CLI arguments natively."""

    @pytest.fixture
    def args_mock(self):
        args = MagicMock()
        args.filename = "default_cli_world.h5"
        return args

    @pytest.fixture
    def world_mock(self):
        return MagicMock()

    @pytest.fixture
    def logger_mock(self):
        return MagicMock()

    def test_serialization_explicitly_disabled(self, world_mock, args_mock, logger_mock):
        """Verify config['serialization']['enabled'] = False aborts export."""
        config = {"serialization": {"enabled": False}}
        
        execute_serialization_block(world_mock, config, args_mock, logger_mock)
        
        # Verify export was never called
        world_mock.export_to_hdf5.assert_not_called()

    def test_serialization_missing_config_file_fails_loud(self, world_mock, args_mock, logger_mock):
        """Verify an enabled export with no config_file is rejected (no scenario-generic default)."""
        config = {} # Missing or empty serialization block -> enabled defaults True, no config_file

        with pytest.raises(ValueError):
            execute_serialization_block(world_mock, config, args_mock, logger_mock)

        world_mock.export_to_hdf5.assert_not_called()

    @patch('os.makedirs')
    def test_serialization_custom_output_directory_generation(self, mock_makedirs, world_mock, args_mock, logger_mock):
        """Verify output_dir invokes os.makedirs and forms the correct joined path."""
        custom_dir = "custom_outputs"
        config = {
            "serialization": {
                "enabled": True,
                "output_dir": custom_dir,
                "config_file": "serializer.yaml"
            }
        }

        execute_serialization_block(world_mock, config, args_mock, logger_mock)

        mock_makedirs.assert_called_once_with(custom_dir, exist_ok=True)
        expected_path = os.path.join(custom_dir, args_mock.filename)
        world_mock.export_to_hdf5.assert_called_once_with(expected_path, config_file="serializer.yaml")

    def test_serialization_custom_filename_priority(self, world_mock, args_mock, logger_mock):
        """Verify YAML filename completely overrides the CLI fallback."""
        custom_name = "specific_world.h5"
        config = {
            "serialization": {
                "enabled": True,
                "filename": custom_name,
                "config_file": "serializer.yaml"
            }
        }

        execute_serialization_block(world_mock, config, args_mock, logger_mock)

        expected_path = os.path.join(".", custom_name)
        world_mock.export_to_hdf5.assert_called_once_with(expected_path, config_file="serializer.yaml")

    def test_serialization_custom_config_link(self, world_mock, args_mock, logger_mock):
        """Verify the custom serializer config gets intercepted and passed as a kwarg."""
        custom_serializer = "custom_serializer.yaml"
        config = {
            "serialization": {
                "enabled": True,
                "config_file": custom_serializer
            }
        }
        
        execute_serialization_block(world_mock, config, args_mock, logger_mock)
        
        expected_path = os.path.join(".", args_mock.filename)
        world_mock.export_to_hdf5.assert_called_once_with(expected_path, config_file=custom_serializer)


@pytest.fixture
def minimal_world():
    """Builds a tiny mocked world in-memory to safely serialize."""
    from may.world import World
    from may.geography import VenueManager
    from may.population import PopulationManager
    from may.config_loader import setup_geography
    
    from may.geography import GeographicalUnit
    from may.geography import Venue
    from may.geography import Geography
    
    geo = Geography(data_dir="", levels=["Region", "MGU", "SGU"])
    
    # Manually inject 1 node per layer so we don't need real CSVs
    root = GeographicalUnit(id=1, name="region_1", level="Region")
    geo.add_geo_unit(root)
    
    mgu = GeographicalUnit(id=2, name="mgu_1", level="MGU", parent=root)
    geo.add_geo_unit(mgu)
    
    sgu = GeographicalUnit(id=3, name="sgu_1", level="SGU", parent=mgu)
    geo.add_geo_unit(sgu)
    
    venues = VenueManager(geo, data_dir="")
    # Manually inject 3 venues (ids are assigned automatically at construction)
    h1 = Venue(name="house_1", venue_type="company", geographical_unit=sgu)
    h2 = Venue(name="house_2", venue_type="company", geographical_unit=sgu)
    s1 = Venue(name="school_1", venue_type="school", geographical_unit=sgu)
    
    venues.add_venue(h1)
    venues.add_venue(h2)
    venues.add_venue(s1)
    
    from may.population import PopulationManager, Person
    
    pop = PopulationManager(geo, data_dir="")
    # Manually inject 10 people
    for i in range(10):
        person = Person(age=20+i, sex="male" if i % 2 == 0 else "female", geographical_unit=sgu)
        pop.add_person(person)
        
    world = World(geography=geo, population=pop, venues=venues)
    return world

class TestWorldHDF5PayloadIntegrity:
    """Verifies the actual binary output of WorldSerializer using the micro-world layout."""

    @pytest.fixture
    def test_export_path(self, tmp_path):
        """Returns a safe temporary path for the HDF5 file."""
        return str(tmp_path / "micro_world_test.h5")

    def test_execute_hdf5_export_and_read_binary(self, minimal_world, test_export_path):
        """Run the actual exporting script on a mocked world component and read the real .h5 response!"""
        
        # 1. Trigger realistic serialization using the internal method natively.
        stats = minimal_world.export_to_hdf5(
            test_export_path, config_file="configs/2021/serialization_config.yaml"
        )

        # Confirm the statistics return trace matches the minimal_world injection size
        assert stats['num_people'] == 10
        assert stats['num_venues'] == 3
        # In the micro world we have 1 Region, 1 SGU, 1 MGU = 3
        assert stats['num_geo_units'] == 3

        # 2. Crack open the written .h5 and check binary integrity!
        assert os.path.exists(test_export_path)
        
        with h5py.File(test_export_path, 'r') as f:
            
            # --- Metadata Assertions ---
            assert f.attrs['num_people'] == 10
            assert f.attrs['num_venues'] == 3
            assert f.attrs['num_geo_units'] == 3
            assert 'serialization_version' in f.attrs
            assert 'MAY_version' in f.attrs
            
            # --- Geography Integrity ---
            assert 'geography' in f
            assert 'ids' in f['geography']
            assert len(f['geography']['ids']) == 3
            assert 'levels' in f['geography']
            assert 'parent_ids' in f['geography']
            
            # --- Population Integrity ---
            assert 'population' in f
            assert 'ids' in f['population']
            assert len(f['population']['ids']) == 10
            assert 'ages' in f['population']
            assert 'sexes' in f['population']
            assert 'geo_unit_ids' in f['population']
            
            # Verify data alignment logic - Ages should roughly correspond to what we defined
            ages_ds = f['population']['ages'][:]
            assert len(ages_ds) == 10

            # --- Registry Enumerations ---
            assert 'metadata' in f
            assert 'registries' in f['metadata']
            assert 'geo_levels' in f['metadata']['registries']
            
            # We defined 3 geo objects (Region, MGU, SGU) - verify they are in the binary!
            # Registries are written as datasets containing the string keys (per `_write_registries`)
            geo_levels_binary = [x.decode('utf-8') for x in f['metadata']['registries']['geo_levels'][:]]
            assert "Region" in geo_levels_binary
            assert "MGU" in geo_levels_binary
            assert "SGU" in geo_levels_binary


def _assert_h5_equivalent(path_a, path_b):
    """Recursively compare two HDF5 files: same tree, datasets and attrs."""
    import numpy as np

    # Deliberate divergences from a baseline export: timestamps, and the
    # inspection attr stamped on datasets written by flush_properties steps.
    ignored_attrs = {'creation_timestamp', 'flushed_by_step'}

    def compare_attrs(obj_a, obj_b, location):
        keys_a = set(obj_a.attrs.keys()) - ignored_attrs
        keys_b = set(obj_b.attrs.keys()) - ignored_attrs
        assert keys_a == keys_b, f"attr keys differ at {location}: {keys_a ^ keys_b}"
        for key in keys_a:
            value_a, value_b = obj_a.attrs[key], obj_b.attrs[key]
            assert np.array_equal(value_a, value_b), f"attr {key} differs at {location}"

    def compare_group(group_a, group_b, location):
        compare_attrs(group_a, group_b, location)
        keys_a = set(group_a.keys())
        keys_b = set(group_b.keys())
        assert keys_a == keys_b, f"children differ at {location}: {keys_a ^ keys_b}"
        for key in keys_a:
            child_a, child_b = group_a[key], group_b[key]
            child_location = f"{location}/{key}"
            assert type(child_a) is type(child_b), f"node type differs at {child_location}"
            if isinstance(child_a, h5py.Group):
                compare_group(child_a, child_b, child_location)
            else:
                compare_attrs(child_a, child_b, child_location)
                data_a, data_b = child_a[()], child_b[()]
                assert np.array_equal(data_a, data_b, equal_nan=(data_a.dtype.kind == 'f')), \
                    f"dataset differs at {child_location}"

    with h5py.File(path_a, 'r') as file_a, h5py.File(path_b, 'r') as file_b:
        compare_group(file_a, file_b, '')


class TestEarlyCanonicalFileEquivalence:
    """Two-stage export (export_core early + export in 'r+') must produce a
    file equivalent to the single-shot export path. See docs/adr/0003."""

    def test_two_stage_export_matches_single_shot(self, minimal_world, tmp_path):
        config_file = "configs/2021/serialization_config.yaml"
        single_shot_path = str(tmp_path / "single_shot.h5")
        two_stage_path = str(tmp_path / "two_stage.h5")

        # Baseline: single-shot export (no canonical file created early)
        stats_single = minimal_world.export_to_hdf5(single_shot_path, config_file=config_file)

        # Two-stage: canonical core file first, then final export reopens it
        minimal_world.create_canonical_file(two_stage_path, config_file=config_file)
        assert os.path.exists(two_stage_path)
        with h5py.File(two_stage_path, 'r') as f:
            assert 'population' in f and 'geography' in f
            assert 'venues' not in f
        stats_two_stage = minimal_world.export_to_hdf5(two_stage_path, config_file=config_file)

        assert stats_single == stats_two_stage
        _assert_h5_equivalent(single_shot_path, two_stage_path)

    def test_export_to_different_path_than_canonical_fails_loud(self, minimal_world, tmp_path):
        config_file = "configs/2021/serialization_config.yaml"
        minimal_world.create_canonical_file(str(tmp_path / "canonical.h5"), config_file=config_file)
        with pytest.raises(ValueError, match="canonical"):
            minimal_world.export_to_hdf5(str(tmp_path / "other.h5"), config_file=config_file)


class TestFlushPropertiesPopulation:
    """Mid-timeline flush of person properties to the canonical file
    (`flush_properties` step, target: population). See tmp/plans phase 2."""

    CONFIG_FILE = "configs/2021/serialization_config.yaml"

    @pytest.fixture
    def world_with_ethnicity(self, minimal_world, tmp_path):
        """Canonical file created; every person carries an 'ethnicity'
        (listed in the 2021 serialization config's population properties)."""
        for person in minimal_world.population.people:
            person.properties["ethnicity"] = f"group_{person.id % 3}"
        canonical_path = str(tmp_path / "canonical.h5")
        minimal_world.create_canonical_file(canonical_path, config_file=self.CONFIG_FILE)
        return minimal_world, canonical_path

    def test_flush_writes_property_column_in_frozen_row_order(self, world_with_ethnicity):
        world, canonical_path = world_with_ethnicity
        expected = {p.id: p.properties["ethnicity"] for p in world.population.people}

        world.flush_properties(target="population", properties=["ethnicity"])

        with h5py.File(canonical_path, 'r') as f:
            dataset = f['population/properties/ethnicity']
            person_ids = f['population/ids'][:]
            values = [v.decode() for v in dataset[:]]
            assert dataset.attrs['flushed_by_step'] == 'flush_properties(target=population)'
            for person_id, value in zip(person_ids, values):
                assert value == expected[person_id]

    def test_flushed_property_is_tombstoned_on_every_person(self, world_with_ethnicity):
        from may.flushing import FlushedPropertyError
        world, _ = world_with_ethnicity

        world.flush_properties(target="population", properties=["ethnicity"])

        for person in world.population.people:
            # Membership stays True (filtering.py checks `attr in properties`
            # before reading) but the read itself must fail loud.
            assert "ethnicity" in person.properties
            with pytest.raises(FlushedPropertyError):
                person.properties["ethnicity"]
            with pytest.raises(FlushedPropertyError):
                person.properties.get("ethnicity")

    def test_final_export_after_flush_matches_no_flush_baseline(self, minimal_world, tmp_path):
        """The whole point of the feature: flushing mid-run must not change
        the on-disk format. Flushed columns are written once at flush time
        and skipped by the final export."""
        for person in minimal_world.population.people:
            person.properties["ethnicity"] = f"group_{person.id % 3}"
            person.properties["work_mode"] = f"mode_{person.id % 2}"

        baseline_path = str(tmp_path / "baseline.h5")
        stats_baseline = minimal_world.export_to_hdf5(baseline_path, config_file=self.CONFIG_FILE)

        flushed_path = str(tmp_path / "flushed.h5")
        minimal_world.create_canonical_file(flushed_path, config_file=self.CONFIG_FILE)
        minimal_world.flush_properties(target="population", properties=["ethnicity", "work_mode"])
        stats_flushed = minimal_world.export_to_hdf5(flushed_path, config_file=self.CONFIG_FILE)

        assert stats_baseline == stats_flushed
        _assert_h5_equivalent(baseline_path, flushed_path)

    def test_unknown_target_fails_loud(self, world_with_ethnicity):
        world, _ = world_with_ethnicity
        with pytest.raises(ValueError, match="target"):
            world.flush_properties(target="peple", properties=["ethnicity"])

    def test_unknown_property_fails_loud(self, world_with_ethnicity):
        """Property not in serialization config would never reach the export."""
        world, _ = world_with_ethnicity
        with pytest.raises(ValueError, match="not_a_property"):
            world.flush_properties(target="population", properties=["not_a_property"])

    def test_empty_properties_list_fails_loud(self, world_with_ethnicity):
        world, _ = world_with_ethnicity
        with pytest.raises(ValueError, match="properties"):
            world.flush_properties(target="population", properties=[])

    def test_flush_without_canonical_file_fails_loud(self, minimal_world):
        with pytest.raises(RuntimeError, match="canonical"):
            minimal_world.flush_properties(target="population", properties=["ethnicity"])

    def test_double_flush_fails_loud(self, world_with_ethnicity):
        world, _ = world_with_ethnicity
        world.flush_properties(target="population", properties=["ethnicity"])
        with pytest.raises(ValueError, match="twice"):
            world.flush_properties(target="population", properties=["ethnicity"])

    def test_cohort_drift_after_flush_fails_loud_at_export(self, world_with_ethnicity):
        """The population cohort must be closed at flush: adding a person
        afterwards would misalign the flushed column with /population/ids."""
        from may.population import Person
        world, canonical_path = world_with_ethnicity
        world.flush_properties(target="population", properties=["ethnicity"])

        sgu = world.geography.units_by_id[3]
        world.population.add_person(Person(age=40, sex="male", geographical_unit=sgu))

        with pytest.raises(RuntimeError, match="cohort"):
            world.export_to_hdf5(canonical_path, config_file=None)

    def test_cohort_drift_between_partial_flushes_fails_loud(self, world_with_ethnicity):
        """Two flush steps may split population properties, but the cohort
        must not change between them: the second flush re-verifies the
        Person.id sequence recorded by the first."""
        from may.population import Person
        world, _ = world_with_ethnicity
        for person in world.population.people:
            person.properties["work_mode"] = f"mode_{person.id % 2}"
        world.flush_properties(target="population", properties=["ethnicity"])

        sgu = world.geography.units_by_id[3]
        world.population.add_person(Person(age=40, sex="male", geographical_unit=sgu))

        with pytest.raises(RuntimeError, match="cohort"):
            world.flush_properties(target="population", properties=["work_mode"])


class TestFlushPropertiesGeoUnits:
    """Mid-timeline flush of geographical unit properties to the canonical
    file (`flush_properties` step, target: geo_units). See tmp/plans phase 3."""

    GEO_PROPERTY = "region_code"

    @pytest.fixture
    def geo_config_file(self, tmp_path):
        """The 2021 serialization config lists no geography properties, so
        derive one that does."""
        import yaml
        with open("configs/2021/serialization_config.yaml") as f:
            config = yaml.safe_load(f)
        config["geography"]["properties"] = [self.GEO_PROPERTY]
        config_path = tmp_path / "serialization_config_geo.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return str(config_path)

    @pytest.fixture
    def world_with_geo_property(self, minimal_world, geo_config_file, tmp_path):
        """Canonical file created; every geo unit carries a 'region_code'."""
        for unit in minimal_world.geography.units_by_id.values():
            unit.properties[self.GEO_PROPERTY] = f"code_{unit.id}"
        canonical_path = str(tmp_path / "canonical.h5")
        minimal_world.create_canonical_file(canonical_path, config_file=geo_config_file)
        return minimal_world, canonical_path

    def test_flush_writes_geo_property_column_in_id_order(self, world_with_geo_property):
        world, canonical_path = world_with_geo_property
        expected = {unit.id: unit.properties[self.GEO_PROPERTY]
                    for unit in world.geography.units_by_id.values()}

        world.flush_properties(target="geo_units", properties=[self.GEO_PROPERTY])

        with h5py.File(canonical_path, 'r') as f:
            dataset = f[f'geography/properties/{self.GEO_PROPERTY}']
            unit_ids = f['geography/ids'][:]
            values = [v.decode() for v in dataset[:]]
            assert dataset.attrs['flushed_by_step'] == 'flush_properties(target=geo_units)'
            for unit_id, value in zip(unit_ids, values):
                assert value == expected[unit_id]

    def test_flushed_property_is_tombstoned_on_every_geo_unit(self, world_with_geo_property):
        from may.flushing import FlushedPropertyError
        world, _ = world_with_geo_property

        world.flush_properties(target="geo_units", properties=[self.GEO_PROPERTY])

        for unit in world.geography.units_by_id.values():
            assert self.GEO_PROPERTY in unit.properties
            with pytest.raises(FlushedPropertyError):
                unit.properties[self.GEO_PROPERTY]
            with pytest.raises(FlushedPropertyError):
                unit.properties.get(self.GEO_PROPERTY)

    def test_final_export_after_flush_matches_no_flush_baseline(self, minimal_world, geo_config_file, tmp_path):
        """Flushing geo properties mid-run must not change the on-disk
        format: flushed columns are written once at flush time and skipped
        by the final export."""
        for unit in minimal_world.geography.units_by_id.values():
            unit.properties[self.GEO_PROPERTY] = f"code_{unit.id}"

        baseline_path = str(tmp_path / "baseline.h5")
        stats_baseline = minimal_world.export_to_hdf5(baseline_path, config_file=geo_config_file)

        flushed_path = str(tmp_path / "flushed.h5")
        minimal_world.create_canonical_file(flushed_path, config_file=geo_config_file)
        minimal_world.flush_properties(target="geo_units", properties=[self.GEO_PROPERTY])
        stats_flushed = minimal_world.export_to_hdf5(flushed_path, config_file=geo_config_file)

        assert stats_baseline == stats_flushed
        _assert_h5_equivalent(baseline_path, flushed_path)

    def test_unknown_geo_property_fails_loud(self, world_with_geo_property):
        """Property not in serialization config would never reach the export."""
        world, _ = world_with_geo_property
        with pytest.raises(ValueError, match="not_a_property"):
            world.flush_properties(target="geo_units", properties=["not_a_property"])

    def test_double_flush_fails_loud(self, world_with_geo_property):
        world, _ = world_with_geo_property
        world.flush_properties(target="geo_units", properties=[self.GEO_PROPERTY])
        with pytest.raises(ValueError, match="twice"):
            world.flush_properties(target="geo_units", properties=[self.GEO_PROPERTY])

    def test_cohort_drift_after_flush_fails_loud_at_export(self, world_with_geo_property):
        """The geography cohort must be closed at flush: adding a unit
        afterwards would misalign the flushed column with /geography/ids."""
        from may.geography import GeographicalUnit
        world, canonical_path = world_with_geo_property
        world.flush_properties(target="geo_units", properties=[self.GEO_PROPERTY])

        mgu = world.geography.units_by_id[2]
        world.geography.add_geo_unit(
            GeographicalUnit(id=4, name="sgu_2", level="SGU", parent=mgu))

        with pytest.raises(RuntimeError, match="cohort"):
            world.export_to_hdf5(canonical_path, config_file=None)

    def test_cohort_drift_between_partial_flushes_fails_loud(self, minimal_world, tmp_path):
        """Two flush steps may split geography properties, but the cohort
        must not change between them: the second flush re-verifies the
        GeographicalUnit.id sequence recorded by the first."""
        import yaml
        from may.geography import GeographicalUnit
        second_property = "second_code"
        with open("configs/2021/serialization_config.yaml") as f:
            config = yaml.safe_load(f)
        config["geography"]["properties"] = [self.GEO_PROPERTY, second_property]
        config_path = tmp_path / "serialization_config_geo_two.yaml"
        config_path.write_text(yaml.safe_dump(config))

        for unit in minimal_world.geography.units_by_id.values():
            unit.properties[self.GEO_PROPERTY] = f"code_{unit.id}"
            unit.properties[second_property] = f"second_{unit.id}"
        canonical_path = str(tmp_path / "canonical.h5")
        minimal_world.create_canonical_file(canonical_path, config_file=str(config_path))
        minimal_world.flush_properties(target="geo_units", properties=[self.GEO_PROPERTY])

        mgu = minimal_world.geography.units_by_id[2]
        minimal_world.geography.add_geo_unit(
            GeographicalUnit(id=4, name="sgu_2", level="SGU", parent=mgu))

        with pytest.raises(RuntimeError, match="cohort"):
            minimal_world.flush_properties(target="geo_units", properties=[second_property])


class TestFlushPropertiesVenues:
    """Mid-timeline flush of a venue-type property cohort to the canonical
    file (`flush_properties` step, target: venues). See tmp/plans phase 4."""

    VENUE_TYPE = "company"

    @pytest.fixture
    def venue_config_file(self, tmp_path):
        """The 2021 serialization config lists no 'company' properties, so
        derive one that does."""
        import yaml
        with open("configs/2021/serialization_config.yaml") as f:
            config = yaml.safe_load(f)
        config["venues"]["types"][self.VENUE_TYPE] = {"properties": ["capacity", "sector"]}
        config_path = tmp_path / "serialization_config_venues.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return str(config_path)

    @pytest.fixture
    def world_with_venue_properties(self, minimal_world, venue_config_file, tmp_path):
        """Canonical file created; companies carry capacity + sector. An
        extra company sits in a lower-id geo unit but is added last, so the
        cohort's geo-sorted order differs from insertion order."""
        from may.geography import Venue
        mgu = minimal_world.geography.units_by_id[2]
        extra_company = Venue(name="company_mgu", venue_type=self.VENUE_TYPE, geographical_unit=mgu)
        minimal_world.venues.add_venue(extra_company)
        companies = [v for v in minimal_world.venues.get_all_venues_list()
                     if v.type == self.VENUE_TYPE]
        for offset, venue in enumerate(companies):
            venue.properties["capacity"] = 10 + offset
            venue.properties["sector"] = f"sector_{venue.id}"
        canonical_path = str(tmp_path / "canonical.h5")
        minimal_world.create_canonical_file(canonical_path, config_file=venue_config_file)
        return minimal_world, canonical_path

    def _cohort_sorted(self, world):
        companies = [v for v in world.venues.get_all_venues_list()
                     if v.type == self.VENUE_TYPE]
        return sorted(companies, key=lambda v: v.geographical_unit.id)

    def test_flush_writes_venue_property_columns_in_cohort_order(self, world_with_venue_properties):
        world, canonical_path = world_with_venue_properties
        cohort = self._cohort_sorted(world)
        expected_capacities = [v.properties["capacity"] for v in cohort]
        expected_sectors = [v.properties["sector"] for v in cohort]

        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                               properties=["capacity", "sector"])

        with h5py.File(canonical_path, 'r') as f:
            type_group = f[f'venues/properties/{self.VENUE_TYPE}']
            assert list(type_group['capacity'][:]) == expected_capacities
            assert [v.decode() for v in type_group['sector'][:]] == expected_sectors
            assert type_group['capacity'].attrs['flushed_by_step'] == \
                f'flush_properties(target=venues, venue_type={self.VENUE_TYPE})'

    def test_flushed_property_is_tombstoned_per_type(self, world_with_venue_properties):
        """Tombstones apply to the flushed cohort only: a school's own
        'capacity' key (different type) must stay readable."""
        from may.flushing import FlushedPropertyError
        world, _ = world_with_venue_properties
        school = next(v for v in world.venues.get_all_venues_list() if v.type == "school")
        school.properties["capacity"] = 99

        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                               properties=["capacity", "sector"])

        for venue in world.venues.get_all_venues_list():
            if venue.type == self.VENUE_TYPE:
                assert "capacity" in venue.properties
                with pytest.raises(FlushedPropertyError):
                    venue.properties["capacity"]
                with pytest.raises(FlushedPropertyError):
                    venue.properties.get("sector")
        assert school.properties["capacity"] == 99

    def test_final_export_after_flush_matches_no_flush_baseline(self, minimal_world, venue_config_file, tmp_path):
        """Flushing a venue cohort mid-run must not change the on-disk
        format: flushed columns are written once at flush time (aligned
        with ranks_in_type) and skipped by the final export."""
        from may.geography import Venue
        mgu = minimal_world.geography.units_by_id[2]
        minimal_world.venues.add_venue(
            Venue(name="company_mgu", venue_type=self.VENUE_TYPE, geographical_unit=mgu))
        companies = [v for v in minimal_world.venues.get_all_venues_list()
                     if v.type == self.VENUE_TYPE]
        for offset, venue in enumerate(companies):
            venue.properties["capacity"] = 10 + offset
            venue.properties["sector"] = f"sector_{venue.id}"

        baseline_path = str(tmp_path / "baseline.h5")
        stats_baseline = minimal_world.export_to_hdf5(baseline_path, config_file=venue_config_file)

        flushed_path = str(tmp_path / "flushed.h5")
        minimal_world.create_canonical_file(flushed_path, config_file=venue_config_file)
        minimal_world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                       properties=["capacity", "sector"])
        stats_flushed = minimal_world.export_to_hdf5(flushed_path, config_file=venue_config_file)

        assert stats_baseline == stats_flushed
        _assert_h5_equivalent(baseline_path, flushed_path)

    def test_cohort_drift_after_flush_fails_loud_at_export(self, world_with_venue_properties, tmp_path):
        """The cohort must be closed at flush: adding a venue of the type
        afterwards would misalign the flushed columns with ranks_in_type."""
        from may.geography import Venue
        world, canonical_path = world_with_venue_properties
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                               properties=["capacity", "sector"])

        sgu = world.geography.units_by_id[3]
        world.venues.add_venue(
            Venue(name="late_company", venue_type=self.VENUE_TYPE, geographical_unit=sgu))

        with pytest.raises(RuntimeError, match="cohort"):
            world.export_to_hdf5(canonical_path, config_file=None)

    def test_cohort_fully_removed_after_flush_fails_loud_at_export(self, world_with_venue_properties):
        """If every venue of a flushed type is later removed, the type has
        no current members and would otherwise be skipped by the per-type
        export loop, leaving orphaned flushed columns on disk undetected."""
        world, canonical_path = world_with_venue_properties
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                               properties=["capacity", "sector"])

        for venue in list(world.venues.get_all_venues_list()):
            if venue.type == self.VENUE_TYPE:
                world.venues.remove_venue(venue)

        with pytest.raises(RuntimeError, match="cohort"):
            world.export_to_hdf5(canonical_path, config_file=None)

    def test_cohort_drift_between_partial_flushes_fails_loud(self, world_with_venue_properties):
        """Two flush steps may split a type's properties, but the cohort
        must not change between them: the second flush re-verifies the
        Venue.id sequence recorded by the first."""
        from may.geography import Venue
        world, _ = world_with_venue_properties
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                               properties=["capacity"])

        sgu = world.geography.units_by_id[3]
        world.venues.add_venue(
            Venue(name="late_company", venue_type=self.VENUE_TYPE, geographical_unit=sgu))

        with pytest.raises(RuntimeError, match="cohort"):
            world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                   properties=["sector"])

    def test_missing_venue_type_fails_loud(self, world_with_venue_properties):
        world, _ = world_with_venue_properties
        with pytest.raises(ValueError, match="venue_type"):
            world.flush_properties(target="venues", properties=["capacity"])

    def test_venue_type_on_other_target_fails_loud(self, world_with_venue_properties):
        world, _ = world_with_venue_properties
        with pytest.raises(ValueError, match="venue_type"):
            world.flush_properties(target="population", properties=["ethnicity"],
                                   venue_type=self.VENUE_TYPE)

    def test_unconfigured_venue_type_fails_loud(self, world_with_venue_properties):
        """Type with no properties in the serialization config: nothing it
        writes could ever reach the export."""
        world, _ = world_with_venue_properties
        with pytest.raises(ValueError, match="not_a_type"):
            world.flush_properties(target="venues", venue_type="not_a_type",
                                   properties=["capacity"])

    def test_unknown_venue_property_fails_loud(self, world_with_venue_properties):
        world, _ = world_with_venue_properties
        with pytest.raises(ValueError, match="not_a_property"):
            world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                   properties=["not_a_property"])

    def test_double_flush_fails_loud(self, world_with_venue_properties):
        world, _ = world_with_venue_properties
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                               properties=["capacity"])
        with pytest.raises(ValueError, match="twice"):
            world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                   properties=["capacity"])
