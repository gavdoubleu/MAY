"""The shipped 2021 household strategy must pin its own `loop_level`.

An unset key defaults to `geography.levels[-1]` (XLGU, 9 England units), which
caps the geo_unit-outer loop at 9 loop units and therefore at 9 Slices — below
the 8-32 worker pool Phase 5 targets. LGU gives 309.
"""
import yaml

STRATEGY_FILE = "configs/2021/households/allocation_strategy.yaml"


def test_shipped_strategy_loops_over_lgu():
    with open(STRATEGY_FILE) as handle:
        strategy = yaml.safe_load(handle)
    assert strategy.get('loop_level') == "LGU"
