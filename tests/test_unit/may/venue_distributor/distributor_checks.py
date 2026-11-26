"""Plain helpers running a distributor class's `check_config`, or the whole `DistributorStep`."""

from may.preflight.config_reader import ConfigReader
from may.preflight.report import PreflightReport
from may.venue_distributor import DistributorStep
from tests.test_unit.may.config_contexts import geography_context

KNOWN_GEOGRAPHY = geography_context(levels=("SGU", "MGU"))
UNKNOWN_GEOGRAPHY = geography_context(levels=None)
KEY_PATH = "timeline.steps[1].config"


def keys(findings):
    """Finding keys with the file prefix taken off; a finding at the naming key stays whole."""
    return [finding.key.split("): ", 1)[1] if "): " in finding.key else finding.key for finding in findings]


def write(tmp_path, text, name="distributor.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def check_config(distributor_class, path, context=UNKNOWN_GEOGRAPHY):
    """The report from loading `path` as a step file and running `distributor_class.check_config` on it."""
    report = PreflightReport()
    reader = ConfigReader({}, "", report).root_config_file(KEY_PATH, path)
    if not reader.failed:
        distributor_class.check_config(reader, context)
    return report


def from_config(path, context=UNKNOWN_GEOGRAPHY):
    return DistributorStep.from_config(path, context, key_path=KEY_PATH)
