"""Loading a step's YAML file for preflight: every failure is one finding, not an exception."""

import yaml

from may.preflight.report import not_mapping_message
from may.utils import path_resolver as pr
from may.utils.yaml_loader import invalid_yaml_message, load_yaml, unreadable_file_message


def load_config_mapping(config_path, report, data_dir=None):
    """The parsed YAML mapping, or `None` with one problem under `config`."""
    try:
        found_path = pr.find_config_file(config_path, data_dir)
    except pr.ConfigFileNotFoundError as error:
        report.problem("config", str(error))
        return None
    try:
        config = load_yaml(found_path)
    except yaml.YAMLError as error:
        report.problem("config", invalid_yaml_message(found_path, error))
        return None
    except (OSError, UnicodeDecodeError) as error:
        report.problem("config", unreadable_file_message(found_path, error))
        return None
    if config is None:
        report.problem("config", f"{found_path} is empty")
        return None
    if not isinstance(config, dict):
        report.problem("config", not_mapping_message(config))
        return None
    return config
