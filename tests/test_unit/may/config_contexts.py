"""Plain helpers building `ConfigContext`s for stage tests."""

from may.geography.geography import Geography, GeographyConfig
from may.preflight.context import ConfigContext

DEFAULT_LEVELS = ("SGU", "MGU", "LGU")


def geography_context(levels=DEFAULT_LEVELS, world_config=None, extra_results=None) -> ConfigContext:
    """A context holding a real `GeographyConfig` with `levels`; `levels=None` = geography failed.

    `extra_results` maps further stage classes to their results, in build order."""
    geography_config = None
    if levels is not None:
        geography_config = GeographyConfig(
            data_dir="data/geography", levels=list(levels), hierarchy_file=[], coord_files={}, filters=None
        )
    results = {Geography: geography_config, **(extra_results or {})}
    return ConfigContext(world_config=world_config, results=results)
