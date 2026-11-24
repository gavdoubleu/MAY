import pytest
from may.geography import Geography
from may.population.population import PopulationManager
from may.population.person import Person

@pytest.fixture
def loaded_geography():
    geo = Geography(data_dir="tests/test_data/micro_world/geography", levels=["SGU", "MGU", "LGU"], hierarchy_file="hierarchy.csv")
    geo.load_from_csv()
    return geo

def test_population_manager_explicit_load(loaded_geography):
    """
    Test loading population from a single explicit CSV file.
    """
    pm = PopulationManager(geography=loaded_geography, data_dir="tests/test_data/micro_world/population")
    column_mapping = {
        "age": "age",
        "sex": "sex",
        "geo_unit": "location"
    }
    
    pm.load_explicit_from_csv("people.csv", column_mapping)

    assert len(pm.people) == 16

    # Check the first person (row 1 in CSV: age=32, sex=f → female, location=SGU_001)
    p1 = pm.get_person(0)
    assert p1 is not None
    assert p1.age == 32
    assert p1.sex == "female"
    assert p1.geographical_unit.name == "SGU_001"

def test_population_manager_explicit_batch_load(loaded_geography):
    """
    Test loading population from multiple MGU-level CSV files in a directory.
    """
    pm = PopulationManager(geography=loaded_geography, data_dir="tests/test_data/micro_world/population/batch")
    column_mapping = {
        "age": "age",
        "sex": "sex",
        "geo_unit": "location"
    }
    
    pm.load_batch_explicit_from_csv(
        data_dir="tests/test_data/micro_world/population/batch",
        column_mapping=column_mapping,
        level="MGU",
    )

    # From MGU_01_pop.csv we have 2 people, from MGU_02_pop.csv we have 2 people
    assert len(pm.people) == 4

    # MGU file load order is non-deterministic (set iteration), so check by sorting
    by_age = sorted(pm.people, key=lambda p: p.age)
    assert by_age[0].age == 5
    assert by_age[0].sex == "unknown"
    assert by_age[0].geographical_unit.name == "SGU_003"

    assert by_age[3].age == 55
    assert by_age[3].sex == "male"
    assert by_age[3].geographical_unit.name == "SGU_001"
    
def test_population_manager_matrix_load(loaded_geography):
    """
    Test loading demographics from matrix CSVs and then generating population.
    """
    pm = PopulationManager(geography=loaded_geography, data_dir="tests/test_data/micro_world/population/matrix")
    
    pm.load_demographics_from_csv(male_file="demographics_male.csv", female_file="demographics_female.csv")
    
    # Verify accurate parsing
    assert pm.precise_demographics["SGU_001"][2]["male"] == 1
    assert pm.precise_demographics["SGU_001"][0]["female"] == 1
    assert pm.precise_demographics["SGU_002"][3]["male"] == 2
    assert pm.precise_demographics["SGU_002"][1]["female"] == 2
    
    # Generate the actual people
    pm.generate_population()
    
    # Total expected based on CSVs:
    # Male: SGU_001 (1), SGU_002 (3), SGU_004 (1) = 5
    # Female: SGU_001 (1), SGU_002 (3), SGU_003 (1) = 5

    assert len(pm.people) == 10
    
    stats = pm.get_statistics()
    assert stats["sex_distribution"]["male"] == 5
    assert stats["sex_distribution"]["female"] == 5


def test_batch_load_partitions_by_the_named_level(tmp_path):
    """Files are named by units of `level`, not by whatever sits at levels[1]."""
    (tmp_path / "hierarchy.csv").write_text("SGU,MGU,LGU\nS1,M1,L1\nS2,M2,L1\n")
    geography = Geography(data_dir=str(tmp_path), levels=["SGU", "MGU", "LGU"], hierarchy_file="hierarchy.csv")
    geography.load_from_csv()
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "L1_pop.csv").write_text("age,sex,location\n30,f,S1\n40,m,S2\n")
    population = PopulationManager(geography=geography, data_dir=str(batch_dir))
    population.load_batch_explicit_from_csv(
        data_dir=str(batch_dir),
        column_mapping={"age": "age", "sex": "sex", "geo_unit": "location"},
        level="LGU",
    )
    assert sorted(person.age for person in population.people) == [30, 40]
