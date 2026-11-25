"""Config stage: the one owner of preflight and from_config for every stage.

A stage inherits `ConfigStage`, declares `display_name`, `config_error` and (block stages)
`config_key` or (step stages whose config is a YAML file path) `config_is_file`, and implements `parse(reader, context)` and `build(parsed, **live)`. The base
reads, nulls a result with problems, raises with one header wording, and logs warnings.
Nothing here names a stage, a key or a domain error.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

from may.preflight.config_reader import ConfigReader
from may.preflight.report import PreflightReport, counted

if TYPE_CHECKING:
    from may.preflight.context import ConfigContext


class ConfigError(Exception):
    """Any config failure; `.report` holds the findings that caused it."""

    def __init__(self, message: str, report: PreflightReport):
        super().__init__(message)
        self.report = report


class PreflightError(ConfigError):
    """The combined preflight failure; message is the combined report."""

    def __init__(self, report: PreflightReport):
        super().__init__(report.format(), report)


def _is_abstract(cls) -> bool:
    """Whether `cls` still has abstract methods. `__abstractmethods__` is not yet set while
    `__init_subclass__` runs, so this repeats `ABCMeta`'s computation."""
    abstract_names = {
        name for name, attribute in vars(cls).items() if getattr(attribute, "__isabstractmethod__", False)
    }
    for base in cls.__bases__:
        for name in getattr(base, "__abstractmethods__", ()):
            if getattr(getattr(cls, name, None), "__isabstractmethod__", False):
                abstract_names.add(name)
    return bool(abstract_names)


class ConfigStage(ABC):
    """Abstract mixin. No state; every method is a classmethod."""

    display_name: str
    config_error: type
    config_key: Optional[str] = None
    # True when the config is a path to a YAML file (a timeline step); the base loads it.
    config_is_file: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if _is_abstract(cls):
            return
        display_name = getattr(cls, "display_name", None)
        if not isinstance(display_name, str) or not display_name:
            raise TypeError(f"config stage {cls.__name__} must set display_name to non-empty text")
        config_error = getattr(cls, "config_error", None)
        if not (isinstance(config_error, type) and issubclass(config_error, ConfigError)):
            raise TypeError(f"config stage {cls.__name__} must set config_error to a ConfigError subclass")

    @classmethod
    @abstractmethod
    def parse(cls, reader: ConfigReader, context: "ConfigContext") -> Any:
        """Read this stage's config through `reader`; return the parsed result."""

    @classmethod
    @abstractmethod
    def build(cls, parsed: Any, **live) -> Any:
        """Build from the parsed result plus live objects; never reads a config."""

    @classmethod
    def preflight(
        cls, config, context: "ConfigContext", key_path: Optional[str] = None
    ) -> Tuple[Any, PreflightReport]:
        """`(parsed result, report)`; the result is `None` when the report has any problem.

        A blank or non-mapping `config` is one problem at the root key path. For a
        `config_is_file` stage, `config` is the file's path and `key_path` the key naming it;
        a file that cannot be loaded is one problem there."""
        root_key_path = key_path if key_path is not None else cls.config_key
        if root_key_path is None:
            raise TypeError(f"{cls.__name__} has no config_key; pass key_path")
        report = PreflightReport()
        root = ConfigReader({}, "", report)
        if cls.config_is_file:
            reader = root.root_config_file(root_key_path, config, give="a path to a YAML file")
        else:
            reader = root.root_reader(root_key_path, config)
        if reader.failed:
            return None, report
        result = cls.parse(reader, context)
        if report.has_problems:
            return None, report
        if result is None:
            raise TypeError(
                f"{cls.__name__}.parse returned None with no problems; None in the context means failed"
            )
        return result, report

    @classmethod
    def from_config(cls, config, context: "ConfigContext", key_path: Optional[str] = None, **live) -> Any:
        """Preflight, raise `config_error` on problems, log warnings, then build."""
        parsed, report = cls.preflight(config, context, key_path=key_path)
        if report.has_problems:
            header = f"{cls.display_name} config cannot be used: {counted(len(report.problems), 'problem')}."
            raise cls.config_error(f"{header}\n\n{report.body()}", report)
        logger = logging.getLogger(cls.__module__)
        for finding in report.warnings:
            logger.warning("%s: %s", finding.key, finding.message)
        return cls.build(parsed, **live)
