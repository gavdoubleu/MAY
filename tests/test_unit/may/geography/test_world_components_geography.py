from may.preflight.context import ConfigContext
import pytest
from may.geography import Geography

def test_from_config_builds_and_loads():
    """
    Test that Geography.from_config initialises a Geography and builds the region hierarchy from the micro-world data.
    """
    # Create a dummy config dict
    config = {
        "geography": {
            "data_dir": "tests/test_data/micro_world/geography",
            "levels": ["SGU", "MGU", "LGU"],
            "hierarchy_file": "hierarchy.csv"
        }
    }
    
    geo = Geography.from_config(config["geography"], ConfigContext())

    assert isinstance(geo, Geography), "from_config must return a Geography instance"
    assert geo.data_dir == "tests/test_data/micro_world/geography", "Geography data_dir not correctly set from config"
    
    # Verify the units were loaded correctly
    all_units = geo.get_all_units()
    assert len(all_units) > 0, "No geographic units were loaded from CSV"
    
    # SGU, MGU, LGU should be available
    lgu_1 = geo.get_unit("LGU_1")
    assert lgu_1 is not None, "Root unit 'LGU_1' was not loaded"
    assert lgu_1.name == "LGU_1"
    assert lgu_1.level == "LGU"
    assert lgu_1.parent is None, "LGU_1 should have no parent (it is the top level)"
    
    # Check intermediate
    mgu_1 = geo.get_unit("MGU_01")
    assert mgu_1 is not None
    assert mgu_1.level == "MGU"
    assert mgu_1.parent.name == "LGU_1"
    
    # Check the fine-grained units
    sgu_001 = geo.get_unit("SGU_001")
    sgu_002 = geo.get_unit("SGU_002")
    
    assert sgu_001 is not None
    assert sgu_002 is not None
    assert sgu_001.level == "SGU"
    assert sgu_002.level == "SGU"
    assert sgu_001.parent.name == "MGU_01"
    assert sgu_002.parent.name == "MGU_01"

    # Ensure get_children works
    mgu_1_children = set(child.name for child in mgu_1.children)
    assert mgu_1_children == {"SGU_001", "SGU_002"}, "MGU_01 should have 001 and 002 as children"
