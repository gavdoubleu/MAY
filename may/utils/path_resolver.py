"""
Module-level singleton for resolving ${config_root}, ${data_root}, ${output_root}
template variables in YAML path strings.

Call init() once at startup (after loading config.yaml). resolve() is a no-op
until init() is called, so bare paths continue to work unchanged.
"""

import os
import re
from typing import List, Optional, Union

_resolver: Optional["PathResolver"] = None


class PathResolver:
    def __init__(self, config_root: str, data_root: Optional[str], output_root: str):
        self._roots = {
            "config_root": config_root,
            "data_root": data_root or "",
            "output_root": output_root,
        }

    def resolve(self, path: str) -> str:
        if not isinstance(path, str):
            return path
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: self._roots.get(m.group(1), m.group(0)),
            path,
        )


def init(config_root: str, data_root: Optional[str], output_root: str) -> None:
    global _resolver
    _resolver = PathResolver(config_root, data_root, output_root)


def resolve(path: str) -> str:
    """Resolve template variables in path. No-op if init() has not been called."""
    if _resolver is None:
        return path
    return _resolver.resolve(path)


def require_config_path(owner, block, hint=""):
    """Resolve a `config:` pointer, refusing to invent one that is not written.

    A default here points every scenario at one scenario's file, and the loader then
    reports a missing file the author never named."""
    if not block.get("config"):
        raise ValueError(f"{owner} must set `config:` to a YAML file{hint}.")
    return resolve(block["config"])


class ConfigFileNotFoundError(FileNotFoundError):
    """Config file not found at any tried location. Message omits the config key; caller adds it."""

    def __init__(self, path: str, tried: List[str]):
        self.path = path
        self.tried = tried
        super().__init__(f"not found; tried {', '.join(tried)}")


def find_config_file(path: Union[str, os.PathLike], data_dir: Optional[str]) -> str:
    """Find a config file: absolute as written; relative in working dir, then under data_dir.

    Returns the path as found. Raises ConfigFileNotFoundError listing every path tried.
    """
    # A `Path` is taken as its text, so `${...}` roots resolve and the tried list joins.
    path = resolve(str(path))
    data_dir = resolve(str(data_dir)) if data_dir else data_dir
    tried = [path]
    if not os.path.isabs(path) and data_dir:
        tried.append(os.path.join(data_dir, path))
    for candidate in tried:
        if os.path.exists(candidate):
            return candidate
    raise ConfigFileNotFoundError(path, tried)


class DataFileSpecError(ValueError):
    """A data-file setting is not a path or a non-empty list of paths."""


class DataFileNotFoundError(FileNotFoundError):
    """Data file(s) missing. Message omits the config key; caller adds it."""

    def __init__(self, missing: List[str]):
        self.missing = missing
        super().__init__(f"not found: {', '.join(missing)}")


def as_path_list(spec, label: str) -> List[str]:
    """A path or list of paths as a list; `DataFileSpecError` naming `label` otherwise."""
    if isinstance(spec, str):
        spec = [spec]
    elif isinstance(spec, (list, tuple)):
        if not spec:
            raise DataFileSpecError(f"{label}: the file list is empty.")
    else:
        raise DataFileSpecError(
            f"{label}: expected a path or a list of paths, got "
            f"{type(spec).__name__} ({spec!r})."
        )
    bad_entries = [path for path in spec if not isinstance(path, str) or not path]
    if bad_entries:
        raise DataFileSpecError(
            f"{label}: every entry must be a non-empty path string; got {bad_entries!r}."
        )
    return list(spec)


def data_file_paths(spec, data_dir: Optional[str], label: str) -> List[str]:
    """Paths of the data files `spec` names: `${...}` resolved, relative ones under `data_dir`.

    Absolute paths are kept as written; relative ones are made absolute. Raises `DataFileSpecError` on a bad shape and
    `DataFileNotFoundError` listing every missing file."""
    if isinstance(spec, os.PathLike):
        spec = os.fspath(spec)
    data_dir = resolve(str(data_dir)) if data_dir else data_dir
    paths = []
    for path in as_path_list(spec, label):
        path = resolve(path)
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.join(data_dir or "", path))
        paths.append(path)
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise DataFileNotFoundError(missing)
    return paths


def check_data_files(report, key: str, spec, data_dir: Optional[str]) -> Optional[List[str]]:
    """`data_file_paths` for `parse_config`: any error is one problem under `key` in `report`.

    `None` on a problem, or if `data_dir` is `None` (its own problem, already reported):
    then only the shape is checked."""
    try:
        if data_dir is None:
            as_path_list(spec, key)
            return None
        return data_file_paths(spec, data_dir, key)
    except DataFileSpecError:
        report.problem(key, f"must be a path or non-empty list of paths; got {spec!r}")
    except DataFileNotFoundError as error:
        report.problem(key, str(error))
    return None
