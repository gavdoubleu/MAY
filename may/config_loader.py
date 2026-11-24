"""
Configuration and command-line argument handling for MAY.
"""

import os
import logging
from may.utils.yaml_loader import load_yaml

logger = logging.getLogger("config_loader")


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the config file

    Returns:
        Dictionary with configuration
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    config = load_yaml(config_path)

    logger.info(f"Loaded configuration from {config_path}")
    return config
