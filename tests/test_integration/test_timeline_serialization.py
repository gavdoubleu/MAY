import pytest
import os
import h5py
from unittest.mock import patch, MagicMock

from tests.test_integration.h5_equivalence import assert_h5_equivalent

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
        assert_h5_equivalent(single_shot_path, two_stage_path)

    def test_export_to_different_path_than_canonical_fails_loud(self, minimal_world, tmp_path):
        config_file = "configs/2021/serialization_config.yaml"
        minimal_world.create_canonical_file(str(tmp_path / "canonical.h5"), config_file=config_file)
        with pytest.raises(ValueError, match="canonical"):
            minimal_world.export_to_hdf5(str(tmp_path / "other.h5"), config_file=config_file)

    def test_export_with_different_config_than_canonical_fails_loud(self, minimal_world, tmp_path):
        canonical_path = str(tmp_path / "canonical.h5")
        minimal_world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        with pytest.raises(ValueError, match="config_file"):
            minimal_world.export_to_hdf5(canonical_path, config_file="configs/1911/serialization_config.yaml")


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

    @pytest.fixture
    def empty_population_world(self):
        """A world with geo units/venues but zero people. Exercises the
        degenerate empty-cohort case for population properties."""
        from may.world import World
        from may.geography import Geography, GeographicalUnit, Venue, VenueManager
        from may.population import PopulationManager

        geo = Geography(data_dir="", levels=["Region", "MGU", "SGU"])
        root = GeographicalUnit(id=1, name="region_1", level="Region")
        geo.add_geo_unit(root)
        mgu = GeographicalUnit(id=2, name="mgu_1", level="MGU", parent=root)
        geo.add_geo_unit(mgu)
        sgu = GeographicalUnit(id=3, name="sgu_1", level="SGU", parent=mgu)
        geo.add_geo_unit(sgu)

        venues = VenueManager(geo, data_dir="")
        venues.add_venue(Venue(name="house_1", venue_type="company", geographical_unit=sgu))

        pop = PopulationManager(geo, data_dir="")
        return World(geography=geo, population=pop, venues=venues)

    def test_properties_group_created_for_empty_people_cohort(
        self, empty_population_world, tmp_path
    ):
        """Pre-refactor behaviour: the /population/properties group exists
        whenever properties are configured, regardless of people count."""
        export_path = str(tmp_path / "empty_population.h5")
        empty_population_world.export_to_hdf5(export_path, config_file=self.CONFIG_FILE)

        with h5py.File(export_path, 'r') as f:
            assert 'properties' in f['population']

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
        assert_h5_equivalent(baseline_path, flushed_path)

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

    def test_all_none_property_is_deferred_not_lost(self, world_with_ethnicity):
        """A property with every value still None at flush time (e.g. its
        distributor step hasn't run yet) must not be marked flushed or
        tombstoned: it stays live so a later flush/export can pick it up
        once it has real values."""
        world, canonical_path = world_with_ethnicity
        # work_mode is configured but nobody has a value for it yet.
        world.flush_properties(target="population", properties=["ethnicity", "work_mode"])

        with h5py.File(canonical_path, 'r') as f:
            assert "work_mode" not in f['population/properties']

        # Not tombstoned: still readable, still None.
        for person in world.population.people:
            assert person.properties.get("work_mode") is None
        for person in world.population.people:
            person.properties["work_mode"] = f"mode_{person.id % 2}"

        # Deferred property can be flushed for real once it has values.
        world.flush_properties(target="population", properties=["work_mode"])

        with h5py.File(canonical_path, 'r') as f:
            dataset = f['population/properties/work_mode']
            person_ids = f['population/ids'][:]
            values = [v.decode() for v in dataset[:]]
            for person_id, value in zip(person_ids, values):
                assert value == f"mode_{person_id % 2}"

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

    def test_retry_after_mid_loop_failure_does_not_raise_on_duplicate_dataset(
        self, world_with_ethnicity
    ):
        """A property whose dataset was written before a later property in
        the same flush call raised must not block a retry: h5py's
        create_dataset raises on a name collision, so a naive retry of the
        identical flush_properties call would previously fail with a
        confusing 'name already exists' error instead of the original
        cause."""
        world, canonical_path = world_with_ethnicity
        for person in world.population.people:
            person.properties["work_mode"] = f"mode_{person.id % 2}"

        original_write = world._serializer._write_property_array
        state = {"raise_once": True}

        def flaky_write(group, prop_name, objects):
            if prop_name == "work_mode" and state["raise_once"]:
                state["raise_once"] = False
                raise IOError("simulated transient failure")
            return original_write(group, prop_name, objects)

        world._serializer._write_property_array = flaky_write

        with pytest.raises(IOError, match="simulated transient failure"):
            world.flush_properties(target="population", properties=["ethnicity", "work_mode"])

        # Retry the identical call: must succeed, not raise on the
        # already-on-disk 'ethnicity' dataset from the failed attempt.
        world.flush_properties(target="population", properties=["ethnicity", "work_mode"])

        with h5py.File(canonical_path, 'r') as f:
            person_ids = f['population/ids'][:]
            ethnicity_values = [v.decode() for v in f['population/properties/ethnicity'][:]]
            work_mode_values = [v.decode() for v in f['population/properties/work_mode'][:]]
            for person_id, ethnicity_value, work_mode_value in zip(
                person_ids, ethnicity_values, work_mode_values
            ):
                assert ethnicity_value == f"group_{person_id % 3}"
                assert work_mode_value == f"mode_{person_id % 2}"


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

    @pytest.fixture
    def empty_geo_world(self):
        """A world with zero geographical units (and consequently zero
        venues/people, which require one). Exercises the degenerate
        empty-cohort case for geography properties."""
        from may.world import World
        from may.geography import Geography, VenueManager
        from may.population import PopulationManager

        geo = Geography(data_dir="", levels=["Region", "MGU", "SGU"])
        venues = VenueManager(geo, data_dir="")
        pop = PopulationManager(geo, data_dir="")
        return World(geography=geo, population=pop, venues=venues)

    def test_properties_group_created_for_empty_geo_unit_cohort(
        self, empty_geo_world, geo_config_file, tmp_path
    ):
        """Pre-refactor behaviour: the /geography/properties group exists
        whenever properties are configured, regardless of unit count."""
        export_path = str(tmp_path / "empty_geo.h5")
        empty_geo_world.export_to_hdf5(export_path, config_file=geo_config_file)

        with h5py.File(export_path, 'r') as f:
            assert 'properties' in f['geography']

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
        assert_h5_equivalent(baseline_path, flushed_path)

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

    def test_properties_subgroup_created_for_venue_type_with_zero_venues(
        self, minimal_world, tmp_path
    ):
        """A configured venue type with no current venues (e.g. never
        added to this world) must still get its per-type properties
        subgroup, matching every other configured type."""
        import yaml
        empty_type = "hospital"
        with open("configs/2021/serialization_config.yaml") as f:
            config = yaml.safe_load(f)
        config["venues"]["types"][empty_type] = {"properties": ["beds"]}
        config_path = tmp_path / "serialization_config_empty_venue_type.yaml"
        config_path.write_text(yaml.safe_dump(config))

        export_path = str(tmp_path / "export.h5")
        minimal_world.export_to_hdf5(export_path, config_file=str(config_path))

        with h5py.File(export_path, 'r') as f:
            assert empty_type in f['venues/properties']

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
        assert_h5_equivalent(baseline_path, flushed_path)

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

    def test_cohort_fully_removed_after_flush_fails_loud_at_export_all_venues_empty(
            self, world_with_venue_properties):
        """Same as the sibling test, but every venue of every type is gone,
        not just the flushed type — this hit the _write_venues early-return
        guard rather than the per-type export loop."""
        world, canonical_path = world_with_venue_properties
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                               properties=["capacity", "sector"])

        for venue in list(world.venues.get_all_venues_list()):
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


class TestFlushPropertiesVenueSubsets:
    """Mid-timeline flush of a venue-type's subset membership to staging
    (`flush_properties` step, target: venues, subsets: true). See
    tmp/plans/streaming-checkpointed-world-export.md phase B. Households
    are the tracer cohort: no member_metadata, biggest resident payload."""

    VENUE_TYPE = "household"

    def test_subsets_on_non_venues_target_fails_loud(self, minimal_world):
        with pytest.raises(ValueError, match="subsets"):
            minimal_world.flush_properties(target="population", properties=[],
                                           subsets=True)

    @pytest.fixture
    def world_with_households(self, minimal_world):
        """Two households, each with the household's default subset
        populated from the existing population."""
        people = list(minimal_world.population.people)
        sgu = minimal_world.geography.units_by_id[3]
        from may.geography import Venue
        household_a = Venue(name="household_a", venue_type=self.VENUE_TYPE, geographical_unit=sgu)
        household_b = Venue(name="household_b", venue_type=self.VENUE_TYPE, geographical_unit=sgu)
        minimal_world.venues.add_venue(household_a)
        minimal_world.venues.add_venue(household_b)
        for person in people[:5]:
            household_a.add_to_subset(person)
        for person in people[5:]:
            household_b.add_to_subset(person)
        return minimal_world

    @pytest.fixture
    def world_with_metadata_subset(self, world_with_households):
        """One household whose subset carries member_metadata, mirroring
        route_distributor.py's per-person row shape (board/alight-style
        numeric fields), to exercise the metadata-staging path households
        otherwise never touch."""
        world = world_with_households
        households = world.venues.get_venues_by_type(self.VENUE_TYPE)
        household_a = households[0]
        for subset in household_a.subsets.values():
            for person in subset.members:
                subset.member_metadata[person.id] = {
                    "leg_idx": 0,
                    "origin_unit_id": 1,
                    "dest_unit_id": 2,
                }
        return world

    def test_final_export_after_metadata_subset_flush_matches_no_flush_baseline(
        self, world_with_metadata_subset, tmp_path
    ):
        """Flushing a subset that carries member_metadata must not drop that
        metadata: the final export must stream staged metadata back in,
        producing output identical to a run that never flushed."""
        world = world_with_metadata_subset

        baseline_path = str(tmp_path / "baseline.h5")
        stats_baseline = world.export_to_hdf5(
            baseline_path, config_file="configs/2021/serialization_config.yaml")

        flushed_path = str(tmp_path / "flushed.h5")
        world.create_canonical_file(flushed_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)
        stats_flushed = world.export_to_hdf5(flushed_path, config_file="configs/2021/serialization_config.yaml")

        assert stats_baseline == stats_flushed
        assert_h5_equivalent(baseline_path, flushed_path)

    def test_flush_gutted_subset_members_raise_on_access(self, world_with_households, tmp_path):
        from may.flushing import FlushedPropertyError
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")

        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)

        households = world.venues.get_venues_by_type(self.VENUE_TYPE)
        for household in households:
            for subset in household.subsets.values():
                assert subset.num_members == 5
                with pytest.raises(FlushedPropertyError):
                    list(subset.members)

    def test_final_export_after_subsets_flush_matches_no_flush_baseline(
        self, world_with_households, tmp_path
    ):
        """Flushing household subset membership mid-run must not change the
        on-disk format: the final export streams staged members back in,
        producing output byte-identical to a run that never flushed."""
        world = world_with_households

        baseline_path = str(tmp_path / "baseline.h5")
        stats_baseline = world.export_to_hdf5(
            baseline_path, config_file="configs/2021/serialization_config.yaml")

        flushed_path = str(tmp_path / "flushed.h5")
        world.create_canonical_file(flushed_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)
        stats_flushed = world.export_to_hdf5(flushed_path, config_file="configs/2021/serialization_config.yaml")

        assert stats_baseline == stats_flushed
        assert_h5_equivalent(baseline_path, flushed_path)

    def test_live_subset_added_after_subsets_flush_fails_loud_at_export(
        self, world_with_households, tmp_path
    ):
        """A household added (or a new subset added to an existing one)
        after the subsets flush would be silently dropped by final export
        (it's not staged, and its live members shadow nothing) — must fail
        loud instead."""
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)

        sgu = world.geography.units_by_id[3]
        from may.geography import Venue
        late_household = Venue(name="late_household", venue_type=self.VENUE_TYPE, geographical_unit=sgu)
        world.venues.add_venue(late_household)
        late_household.add_to_subset(next(iter(world.population.people)))

        with pytest.raises(RuntimeError, match="household"):
            world.export_to_hdf5(canonical_path, config_file=None)

    def test_repeat_flush_with_same_keys_is_a_no_op(self, world_with_households, tmp_path):
        """Re-running the same subsets-flush step for a venue_type that's
        already fully flushed (e.g. crash-recovery retry) must skip
        cleanly, not crash on the now-gutted subsets."""
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)

        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)  # no raise

        world.export_to_hdf5(canonical_path, config_file=None)

    def test_repeat_flush_after_new_subset_added_fails_loud(self, world_with_households, tmp_path):
        """A subset added to this venue_type between two flush attempts
        leaves a mix of live and already-gutted subsets: must raise a
        clear error, not the generic gutted-members access error."""
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)

        sgu = world.geography.units_by_id[3]
        from may.geography import Venue
        late_household = Venue(name="late_household", venue_type=self.VENUE_TYPE, geographical_unit=sgu)
        world.venues.add_venue(late_household)
        late_household.add_to_subset(next(iter(world.population.people)))

        with pytest.raises(RuntimeError, match=self.VENUE_TYPE):
            world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                                   subsets=True)

    def test_staging_dir_deleted_after_successful_export(self, world_with_households, tmp_path):
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)
        staging_dir = tmp_path / "staging"
        assert staging_dir.exists()

        world.export_to_hdf5(canonical_path, config_file=None)

        assert not staging_dir.exists()

    def test_zero_subset_venue_type_flush_skips_with_warning(self, minimal_world, tmp_path, caplog):
        """A configured venue_type with no subsets anywhere: skip + warn,
        no staging file, nothing gutted, export unaffected (mirrors the
        all-None property-flush policy)."""
        world = minimal_world
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")

        with caplog.at_level("WARNING", logger="world_serializer"):
            world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                                   subsets=True)

        assert "No subsets to flush" in caplog.text
        assert not (tmp_path / "staging").exists()
        world.export_to_hdf5(canonical_path, config_file=None)  # drift check must be a no-op

    def test_crash_mid_staging_leaves_members_live_then_retry_succeeds(
        self, world_with_households, tmp_path
    ):
        """Verify-then-gut's failure side: a crash during the staging write
        must leave every subset live (nothing gutted), and an identical
        retry must succeed."""
        from may.flushing import FlushedPropertyError
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")

        with patch.object(world._serializer, "_append_staging_dataset",
                          side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                       properties=[], subsets=True)

        households = world.venues.get_venues_by_type(self.VENUE_TYPE)
        for household in households:
            for subset in household.subsets.values():
                assert len(list(subset.members)) == 5  # still live, not gutted

        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)  # retry, no raise

        for household in households:
            for subset in household.subsets.values():
                with pytest.raises(FlushedPropertyError):
                    list(subset.members)

        world.export_to_hdf5(canonical_path, config_file=None)

    def test_crash_mid_merge_keeps_staging_warns_and_closes_readers(
        self, world_with_households, tmp_path, caplog
    ):
        """A crash during the final export merge must keep staging/ for
        inspection/retry, say so with a warning, and close the staging
        readers (a retry re-inits the reader cache, orphaning open ones)."""
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)

        # _write_registries runs after the staged members have been merged
        # (readers open), so raising there simulates a crash mid-merge.
        with patch.object(world._serializer, "_write_registries",
                          side_effect=RuntimeError("merge crash")):
            with caplog.at_level("WARNING", logger="world_serializer"):
                with pytest.raises(RuntimeError, match="merge crash"):
                    world.export_to_hdf5(canonical_path, config_file=None)

        assert (tmp_path / "staging" / f"{self.VENUE_TYPE}.h5").exists()
        assert "kept for inspection/retry" in caplog.text
        assert world._serializer._subset_staging_readers == {}

    def test_second_export_after_success_fails_loud(self, world_with_households, tmp_path):
        """A successful export deletes staging/ but leaves husks behind: a
        repeat export cannot stream the flushed subsets back, so it must
        raise a clear error up front rather than an OSError mid-export."""
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)
        world.export_to_hdf5(canonical_path, config_file=None)

        with pytest.raises(RuntimeError, match="exported once"):
            world.export_to_hdf5(canonical_path, config_file=None)

    def test_repeat_subsets_flush_after_export_fails_loud(self, world_with_households, tmp_path):
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)
        world.export_to_hdf5(canonical_path, config_file=None)

        with pytest.raises(RuntimeError, match="cannot repeat"):
            world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                                   subsets=True)

    def test_member_metadata_reassigned_after_flush_fails_loud_at_export(
        self, world_with_metadata_subset, tmp_path
    ):
        """Plain attribute reassignment of member_metadata on a gutted
        subset bypasses the sentinel; export would write the staged
        metadata and silently drop the new rows — must fail loud instead."""
        world = world_with_metadata_subset
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)

        households = world.venues.get_venues_by_type(self.VENUE_TYPE)
        reassigned_subset = next(iter(households[0].subsets.values()))
        reassigned_subset.member_metadata = {}

        with pytest.raises(RuntimeError, match="reassigned"):
            world.export_to_hdf5(canonical_path, config_file=None)

    @pytest.fixture
    def world_with_two_metadata_types(self, world_with_metadata_subset):
        """Adds a live school subset carrying member_metadata whose field
        names partially overlap the household metadata, so an export mixing
        gutted (staged) and live metadata carriers exercises the
        field-union merge order."""
        world = world_with_metadata_subset
        school = world.venues.get_venues_by_type("school")[0]
        for person in list(world.population.people)[:3]:
            school.add_to_subset(person)
        for subset in school.subsets.values():
            for person in subset.members:
                subset.member_metadata[person.id] = {
                    "leg_idx": 1,
                    "t_board_min": 12,
                    "t_alight_min": 30,
                }
        return world

    def test_mixed_gutted_and_live_metadata_types_match_no_flush_baseline(
        self, world_with_two_metadata_types, tmp_path
    ):
        """Flushing one metadata-carrying venue type while another stays
        live must still produce a membership_metadata side-table identical
        to a run that never flushed (row order and field-union order)."""
        world = world_with_two_metadata_types

        baseline_path = str(tmp_path / "baseline.h5")
        stats_baseline = world.export_to_hdf5(
            baseline_path, config_file="configs/2021/serialization_config.yaml")

        flushed_path = str(tmp_path / "flushed.h5")
        world.create_canonical_file(flushed_path, config_file="configs/2021/serialization_config.yaml")
        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE, properties=[],
                               subsets=True)
        stats_flushed = world.export_to_hdf5(flushed_path, config_file="configs/2021/serialization_config.yaml")

        assert stats_baseline == stats_flushed
        assert_h5_equivalent(baseline_path, flushed_path)


class TestFlushPropertiesCombinedRetrySafety:
    """A single flush_properties(properties=[...], subsets=True) call must
    be safe to retry identically after a partial failure: subsets runs
    before properties, and properties are validated up front, so neither
    ordering leaves a retry stuck on CohortFlushTracker's "flushed twice"
    check (see tmp/plans/fix-partial-flush-no-rollback.md)."""

    VENUE_TYPE = "household"

    @pytest.fixture
    def world_with_households(self, minimal_world):
        people = list(minimal_world.population.people)
        sgu = minimal_world.geography.units_by_id[3]
        from may.geography import Venue
        household_a = Venue(name="household_a", venue_type=self.VENUE_TYPE, geographical_unit=sgu)
        household_b = Venue(name="household_b", venue_type=self.VENUE_TYPE, geographical_unit=sgu)
        minimal_world.venues.add_venue(household_a)
        minimal_world.venues.add_venue(household_b)
        for offset, household in enumerate((household_a, household_b)):
            household.properties["capacity"] = 5 + offset
        for person in people[:5]:
            household_a.add_to_subset(person)
        for person in people[5:]:
            household_b.add_to_subset(person)
        return minimal_world

    def test_retry_after_subsets_failure_leaves_properties_untouched_then_succeeds(
        self, world_with_households, tmp_path
    ):
        from may.flushing import FlushedPropertyError
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")

        households = world.venues.get_venues_by_type(self.VENUE_TYPE)

        with patch.object(
            world._serializer, "flush_venue_subsets",
            side_effect=RuntimeError("staging write failed"),
        ):
            with pytest.raises(RuntimeError, match="staging write failed"):
                world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                       properties=["capacity"], subsets=True)

        # Properties must not have been flushed by the failed call.
        for household in households:
            assert household.properties["capacity"] is not None

        world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                               properties=["capacity"], subsets=True)  # retry, no raise

        for household in households:
            with pytest.raises(FlushedPropertyError):
                household.properties["capacity"]
            for subset in household.subsets.values():
                with pytest.raises(FlushedPropertyError):
                    list(subset.members)

    def test_retry_after_properties_failure_skips_already_flushed_subsets(
        self, world_with_households, tmp_path, caplog
    ):
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")

        with patch.object(
            world._serializer, "flush_venue_properties",
            side_effect=RuntimeError("write failed"),
        ):
            with pytest.raises(RuntimeError, match="write failed"):
                world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                       properties=["capacity"], subsets=True)

        # Subsets succeeded before properties raised.
        households = world.venues.get_venues_by_type(self.VENUE_TYPE)
        from may.flushing import FlushedPropertyError
        for household in households:
            for subset in household.subsets.values():
                with pytest.raises(FlushedPropertyError):
                    list(subset.members)

        staging_path = tmp_path / "staging" / f"{self.VENUE_TYPE}.h5"
        mtime_before_retry = staging_path.stat().st_mtime_ns

        with caplog.at_level("INFO", logger="world_serializer"):
            world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                   properties=["capacity"], subsets=True)  # retry, no raise

        assert "already flushed with same keys; skipping" in caplog.text
        assert staging_path.stat().st_mtime_ns == mtime_before_retry

        for household in households:
            with pytest.raises(FlushedPropertyError):
                household.properties["capacity"]

    def test_unknown_property_with_subsets_fails_before_any_staging_write(
        self, world_with_households, tmp_path
    ):
        world = world_with_households
        canonical_path = str(tmp_path / "canonical.h5")
        world.create_canonical_file(canonical_path, config_file="configs/2021/serialization_config.yaml")

        with pytest.raises(ValueError, match="not_a_property"):
            world.flush_properties(target="venues", venue_type=self.VENUE_TYPE,
                                   properties=["not_a_property"], subsets=True)

        assert not (tmp_path / "staging").exists()

        households = world.venues.get_venues_by_type(self.VENUE_TYPE)
        for household in households:
            for subset in household.subsets.values():
                assert list(subset.members)  # never gutted
