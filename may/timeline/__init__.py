"""The build timeline: which step types exist, checking a timeline, and running its steps in order."""
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from may.attribute_assignment import AttributeAssignmentConfig, AttributeAssignmentError
from may.preflight.config_stage import ConfigStage
from may.preflight.context import ConfigContext
from may.preflight.report import PreflightReport, not_list_message, not_mapping_message
from may.residence.allocation_strategy import AllocationStrategy
from may.residence.household_distributor import HouseholdError
from may.utils import build_profile as bp
from may.utils import path_resolver as pr
from may.venue_child_creator import VenueChildCreator, VenueChildCreatorError
from may.venue_distributor import DistributorConfigError, distributor_from_yaml, distributor_preflight

logger = logging.getLogger(__name__)


class TimelineError(Exception):
    """A timeline step failed. Already logged; the cause is chained."""


@dataclass(frozen=True)
class TimelineStep:
    """What the timeline needs to check and run one step type.

    ``step_class`` is a ``ConfigStage`` whose config is the step file, or (not yet
    migrated) has ``preflight(config_path, context) -> PreflightReport`` and
    ``from_config(config_path, world_config)``; ``run`` names the method of the built
    object that runs the step on the world. ``expected_errors`` are failures the entry
    point reports in its own words, so they are logged without a traceback.
    ``config_hint`` is appended to the missing-``config:`` message where the step name
    does not say what to point at."""
    step_class: Any
    run: str
    expected_errors: tuple = ()
    config_hint: str = ""


# The distributor step dispatches on `distributor_type`, so its owner is a pair of
# module functions rather than one class.
_DISTRIBUTOR = SimpleNamespace(preflight=distributor_preflight, from_config=distributor_from_yaml)


TIMELINE_STEPS = {
    "residence_allocation": TimelineStep(
        step_class=AllocationStrategy,
        run="allocate",
        expected_errors=(HouseholdError,),
        config_hint=" (e.g. configs/<scenario>/households/allocation_strategy.yaml)",
    ),
    "attribute": TimelineStep(
        step_class=AttributeAssignmentConfig,
        run="assign_attributes",
        expected_errors=(AttributeAssignmentError,),
    ),
    "distributor": TimelineStep(
        step_class=_DISTRIBUTOR,
        run="allocate",
        expected_errors=(DistributorConfigError,),
    ),
    "child_creator": TimelineStep(
        step_class=VenueChildCreator,
        run="create_children",
        expected_errors=(VenueChildCreatorError,),
    ),
}


@dataclass(frozen=True)
class _ParsedStep:
    position_key: str
    prefix: str
    step_type: str
    config_path: str

    @property
    def config_key_path(self):
        """Where a config stage step keys its findings: the key naming its file."""
        return f"{self.position_key}.config"


def _is_config_stage(step_class):
    return isinstance(step_class, type) and issubclass(step_class, ConfigStage)


def _parse_timeline(timeline_block, households_block_present):
    """The runnable steps of a timeline block, with a report of its structure problems.

    The one owner of the timeline's structure checks: `preflight` reports them and
    `run_timeline` raises on them."""
    report = PreflightReport()
    if timeline_block is None:
        report.problem("timeline", (
            "not set; every config must define `timeline.enabled: true` with "
            "`timeline.steps`, including a `residence_allocation` step when "
            "households are enabled"
        ))
        return [], report
    if not isinstance(timeline_block, dict):
        report.problem("timeline", not_mapping_message(timeline_block))
        return [], report
    if not timeline_block.get("enabled", False):
        report.problem("timeline.enabled", "must be true; every config builds its world through the timeline")
    raw_steps = timeline_block.get("steps")
    if raw_steps is None:
        report.problem("timeline.steps", "not set; list at least one step")
        return [], report
    if not isinstance(raw_steps, list):
        report.problem("timeline.steps", not_list_message(raw_steps))
        return [], report
    if not raw_steps:
        report.problem("timeline.steps", "is empty; list at least one step")

    steps = []
    for position, raw_step in enumerate(raw_steps, start=1):
        parsed_step = _parse_step(position, raw_step, report)
        if parsed_step is not None:
            steps.append(parsed_step)

    # A households: block only wires data for the residence_allocation step, so
    # a block with no step is almost always a mistake.
    step_types = [step.get("type") for step in raw_steps if isinstance(step, dict)]
    if households_block_present and "residence_allocation" not in step_types:
        report.warning("households", (
            "configured but the timeline has no 'residence_allocation' step; "
            "households will NOT be allocated"
        ))
    return steps, report


def _parse_step(position, raw_step, report):
    """One step, or `None` with its problems in `report`."""
    position_key = f"timeline.steps[{position}]"
    if not isinstance(raw_step, dict):
        report.problem(position_key, f"must be a mapping with `type:` and `config:`; got {raw_step!r}")
        return None
    config_file = raw_step.get("config")
    if not isinstance(config_file, str):
        config_file = None
    prefix = f"{position_key} ({config_file})" if config_file else position_key
    step_type = raw_step.get("type")
    known_type = isinstance(step_type, str) and step_type in TIMELINE_STEPS
    if not known_type:
        report.problem(f"{prefix}: type", (
            f"unknown step type {step_type!r}; expected one of {sorted(TIMELINE_STEPS)}"
        ))
    if not config_file:
        hint = TIMELINE_STEPS[step_type].config_hint if known_type else ""
        report.problem(f"{position_key}: config", f"must be set to a YAML file{hint}")
    if not (known_type and config_file):
        return None
    return _ParsedStep(position_key, prefix, step_type, pr.resolve(config_file))


def preflight(timeline_block, context):
    """Check a timeline block and every step's config, without building anything.

    Step findings are keyed `timeline.steps[N] (<file>): <key>`, N counted from 1; a
    config stage step's `timeline.steps[N].config (<found file>): <key>`."""
    steps, report = _parse_timeline(timeline_block, context.households_block_present)
    step_reports = [_preflight_step(step, context) for step in steps]
    return PreflightReport.merge(report, *step_reports)


def _preflight_step(step, context):
    step_class = TIMELINE_STEPS[step.step_type].step_class
    if _is_config_stage(step_class):
        # Config stage steps ignore the old context until the timeline stage passes theirs.
        _, report = step_class.preflight(step.config_path, ConfigContext(), key_path=step.config_key_path)
        return report
    return step_class.preflight(step.config_path, context).under(step.prefix)


def step_label(step_type, step_config):
    """Name a timeline step after the config file it runs, so twelve
    distributors read as twelve stages rather than one."""
    if not step_config:
        return step_type
    return f"{step_type}:{os.path.splitext(os.path.basename(step_config))[0]}"


def run_timeline(world, world_config):
    """Run the world config's timeline steps in the order they are written.

    Raises `TimelineError` on the same structure problems `preflight` reports, so a
    build that skipped preflight still stops with the same named problem."""
    steps, report = _parse_timeline(
        world_config.get("timeline"), bool(world_config.get("households"))
    )
    for warning in report.warnings:
        logger.warning(f"{warning.key}: {warning.message}")
    try:
        report.raise_if_problems(TimelineError, "The timeline cannot run.")
    except TimelineError as error:
        logger.error(str(error))
        raise

    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION TIMELINE")
    logger.info("=" * 60)

    for step in steps:
        # Steps of one type are told apart by their config file, since a
        # timeline runs a dozen distributors through the same code path.
        with bp.stage(step_label(step.step_type, step.config_path), step.step_type, detail=step.config_path):
            logger.info("")
            logger.info(f"[{step.step_type.replace('_', ' ').upper()}] {step.config_path}")
            _run_step(TIMELINE_STEPS[step.step_type], step, world, world_config)


def _build_step(entry, step, world_config):
    if _is_config_stage(entry.step_class):
        return entry.step_class.from_config(
            step.config_path, ConfigContext(world_config=world_config), key_path=step.config_key_path
        )
    return entry.step_class.from_config(step.config_path, world_config)


def _run_step(entry, step, world, world_config):
    """Build and run one step, raising `TimelineError` on failure. An expected error
    already says what went wrong, so only an unexpected one carries a traceback."""
    step_type, config_path = step.step_type, step.config_path
    try:
        built_step = _build_step(entry, step, world_config)
        getattr(built_step, entry.run)(world)
    except entry.expected_errors as error:
        message = f"Timeline step {step_type!r} failed for {config_path}: {error}"
        logger.error(message)
        raise TimelineError(message) from error
    except Exception as error:
        message = f"Timeline step {step_type!r} failed for {config_path}: {error}"
        logger.exception(message)
        raise TimelineError(message) from error
