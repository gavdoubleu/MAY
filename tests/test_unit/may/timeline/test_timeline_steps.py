"""
The step registry names the four step types; each step is labelled by the config file
it runs. Structure checks are in `test_timeline_preflight.py`.
"""
import pathlib
import subprocess
import sys

import pytest

from may import timeline


def test_the_step_vocabulary_is_the_four_dispatched_types():
    assert set(timeline.TIMELINE_STEPS) == {
        "residence_allocation", "attribute", "distributor", "child_creator",
    }


def test_a_step_is_labelled_by_its_type_and_config_stem():
    """Twelve distributors read as twelve build-profile stages rather than one."""
    assert timeline.step_label("distributor", "/a/b/pubs.yaml") == "distributor:pubs"


@pytest.mark.parametrize("absent", [None, ""])
def test_a_step_without_config_is_labelled_by_its_type(absent):
    assert timeline.step_label("romantic", absent) == "romantic"


def test_importing_the_timeline_seeds_nothing_and_configures_no_logging():
    """`create_world` seeds RNGs and configures logging at import; the timeline must not."""
    probe = (
        "import logging, numpy as np\n"
        "np.random.seed(12345)\n"
        "before = np.random.get_state()[1].copy()\n"
        "import may.timeline\n"
        "assert (np.random.get_state()[1] == before).all(), 'numpy RNG reseeded'\n"
        "assert not logging.getLogger().handlers, 'root logger configured'\n"
        "assert logging.getLogger().level == logging.WARNING, 'root level changed'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=pathlib.Path(__file__).parents[4],
    )

    assert result.returncode == 0, result.stderr
