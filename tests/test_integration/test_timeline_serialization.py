import pytest
import os
import h5py

from may.serialization import SerialisationConfig

SCHEMA_2021 = "configs/2021/serialization_config.yaml"


@pytest.fixture
def minimal_world():
    """Builds a tiny mocked world in-memory to safely serialize."""
    from may.world import World
    from may.geography import VenueManager
    from may.population import PopulationManager
    
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
            test_export_path, SerialisationConfig.from_file(SCHEMA_2021)
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


class TestExportRejectsUnwritableProperties:
    """
    A property whose values span more than one column type is a build-time
    defect the export is the first step to see. It names every such property in
    one run and removes the partial file it wrote.
    """

    def test_all_offending_properties_are_named_in_one_error(
        self, minimal_world, tmp_path
    ):
        people = minimal_world.population.get_all_people()
        # Two properties, each holding the mix of types that stacked files and
        # runtime assignment can produce.
        for index, person in enumerate(people):
            person.properties["ethnicity"] = "white" if index else 5
            person.properties["work_mode"] = 1.5 if index else 3

        export_path = str(tmp_path / "rejected.h5")
        with pytest.raises(Exception) as excinfo:
            minimal_world.export_to_hdf5(
                export_path, SerialisationConfig.from_file(SCHEMA_2021)
            )

        message = str(excinfo.value)
        assert "ethnicity" in message, "the first offending property is named"
        assert "work_mode" in message, "the export continues past the first"
        assert not os.path.exists(export_path), "the partial file is removed"

    def test_a_clean_world_still_exports(self, minimal_world, tmp_path):
        export_path = str(tmp_path / "clean.h5")
        for person in minimal_world.population.get_all_people():
            person.properties["ethnicity"] = "white"
        stats = minimal_world.export_to_hdf5(
            export_path, SerialisationConfig.from_file(SCHEMA_2021)
        )
        assert stats["num_people"] == 10
        assert os.path.exists(export_path)
