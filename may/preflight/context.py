"""What earlier config stages parsed, for later stages' checks and the build.

`ConfigContext` replaces `PreflightContext`, which stays until every stage has moved."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, FrozenSet, Iterator, Mapping, Optional

from may.preflight.config_stage import ConfigStage


class StageNotParsed(KeyError):
    """A stage looked up in a context it was never added to: a programmer error."""

    def __init__(self, stage: type):
        super().__init__(stage)
        self.stage = stage

    def __str__(self) -> str:
        return f"{self.stage.__name__} has not been parsed; it is not in this config context"


def _check_stage_key(stage) -> None:
    if not (isinstance(stage, type) and issubclass(stage, ConfigStage) and stage is not ConfigStage):
        raise TypeError(f"config context keys are config stage classes; got {stage!r}")


class ConfigContext:
    """Immutable, build-ordered map from config stage class to its parsed result.

    Present + `None` = the stage failed; not present = never parsed (`StageNotParsed`).
    Lookup is by exact class. `world_config` is the raw world config, read-only."""

    __slots__ = ("_results", "_world_config")

    def __init__(self, world_config: Optional[Mapping] = None, results: Optional[Mapping] = None):
        checked_results = {}
        for stage, result in (results or {}).items():
            _check_stage_key(stage)
            checked_results[stage] = result
        object.__setattr__(self, "_results", checked_results)
        object.__setattr__(self, "_world_config", MappingProxyType(dict(world_config or {})))

    def __setattr__(self, name, value):
        raise AttributeError("ConfigContext is immutable; use with_result")

    def __delattr__(self, name):
        raise AttributeError("ConfigContext is immutable")

    @property
    def world_config(self) -> Mapping:
        return self._world_config

    def __getitem__(self, stage: type) -> Any:
        _check_stage_key(stage)
        if stage not in self._results:
            raise StageNotParsed(stage)
        return self._results[stage]

    def __contains__(self, stage) -> bool:
        return stage in self._results

    def stages(self) -> Iterator[type]:
        """Stage classes in build order."""
        return iter(tuple(self._results))

    def with_result(self, stage: type, result: Any) -> "ConfigContext":
        """New context with `stage` appended; this one is unchanged."""
        _check_stage_key(stage)
        if stage in self._results:
            raise ValueError(f"{stage.__name__} is already in this config context")
        return ConfigContext(world_config=self._world_config, results={**self._results, stage: result})


@dataclass(frozen=True)
class PreflightContext:
    """Immutable. A fact is `None` when the block that supplies it failed, and the
    checks that depend on it are skipped, so one root cause gives one finding.

    `geo_levels` is the levels `Geography` parsed. `households` is the
    settings `HouseholdDistributor` parsed, not the raw block. `households_block_present`
    is whether a `households:` block is written at all, parsed or not."""
    geo_levels: Optional[FrozenSet[str]] = None
    households: Optional[Any] = None
    households_block_present: bool = False


def check_geo_level(key, geo_level, context, report):
    """A geo level, when set, must be a name; a known one once the geography is known."""
    if geo_level is None:
        return
    if not isinstance(geo_level, str):
        report.problem(key, f"must be a geo level name; got {geo_level!r}")
        return
    if context.geo_levels is not None and geo_level not in context.geo_levels:
        report.problem(key, f"unknown geo level {geo_level!r}; expected one of {sorted(context.geo_levels)}")
