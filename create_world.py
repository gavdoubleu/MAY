# -*- coding: utf-8 -*-
import cProfile
import os
import logging
import sys
import numpy as np
import numba as nb
import yaml
from may.geography import Geography, GeographyError, VenueManager, VenueError
from may.population import PopulationManager, PopulationError
from may.serialization import SerialisationError, WorldSerializer
from may.world import World
from may.social_networks import SocialNetworkBuilder
from may.preflight.preflight_coordinator import run_preflight
from may.preflight.report import PreflightError
from may.utils import path_resolver as pr
from may.utils import build_profile as bp
from may.timeline import TimelineError, run_timeline, step_label
#from debug_scripts.check_multiple_jobs import analyze_multiple_jobs

logger = logging.getLogger("create_world")

# Windows consoles default to cp1252, which cannot encode the non-ASCII
# characters used in log messages. Force UTF-8 on the log stream.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Suppress numexpr logging
logging.getLogger('numexpr').setLevel(logging.WARNING)

def set_random_seed(seed=999):
    """
    Sets global seeds for testing in numpy and numbaised numpy.
    """

    @nb.njit(cache=True)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    np.random.seed(seed)
    set_seed_numba(seed)
    return

set_random_seed(0)


def parse_args(argv=None):
    """Parse the CLI. Shared with the entry point below, which needs to know
    whether cProfile was asked for before the build starts."""
    import argparse
    parser = argparse.ArgumentParser(description="Create a simulated world from configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/2021/config.yaml",
        help="Path to configuration YAML file (default: configs/2021/config.yaml)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Output HDF5 filename. Overrides serialization.filename in config. Defaults to world_state.h5 if neither is set."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Also record function-level cProfile data. Slows the build and shifts the timings it reports, so it is off by default."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check the config and exit (0 if clean, 1 on problems) without building or writing anything."
    )
    return parser.parse_args(argv)


def main(args=None):
    """
    Main entry point for world creation.
    """

    logger.info("=" * 60)
    logger.info("MAY - World Creation")
    logger.info("=" * 60)

    if args is None:
        args = parse_args()

    logger.info(f"Loading configuration from: {args.config}")
    with open(args.config, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    # Initialise path resolver from roots declared in config.yaml
    from pathlib import Path as _Path
    _config_yaml_dir = str(_Path(args.config).resolve().parent)
    pr.init(
        config_root=config.get("config_root", _config_yaml_dir),
        data_root=config.get("data_root", None),
        output_root=config.get("output_root", str(_Path.cwd() / "output")),
    )

    try:
        run_preflight(config)
    except PreflightError:
        logger.error("Preflight failed; fix the config problems above and rerun.")
        sys.exit(1)

    if getattr(args, "preflight_only", False):
        logger.info("--preflight-only given; stopping before the build.")
        return None

    try:
        serialisation = WorldSerializer.from_config(config.get("serialization") or {})
    except SerialisationError as e:
        logger.error(f"Serialisation config failed: {e}")
        sys.exit(1)

    # Every build records what its stages cost. The profile lands next to the
    # world this run produces.
    bp.start(serialisation.output_dir, args.config, cprofile_enabled=getattr(args, "profile", False))

    # Setup geography from config and command-line arguments
    with bp.stage("geography_load", "pipeline"):
        try:
            geo = Geography.from_config(config.get("geography") or {})
        except GeographyError as e:
            logger.error(f"Geography loading failed: {e}")
            sys.exit(1)
        bp.bind(geography=geo)

    # Load venues
    logger.info("")
    logger.info("Loading venues...")
    with bp.stage("venues_load", "pipeline"):
        try:
            venues = VenueManager.from_config(config.get("venues") or {}, geo)
        except VenueError as e:
            logger.error(f"Venue loading failed: {e}")
            sys.exit(1)
        bp.bind(venues=venues)

    # Load population
    logger.info("")
    logger.info("Loading population...")
    with bp.stage("population_load", "pipeline"):
        try:
            population = PopulationManager.from_config(config.get("population") or {}, geo)
        except PopulationError as e:
            logger.error(f"Population loading failed: {e}")
            sys.exit(1)
        bp.bind(population=population)

    # Households are allocated by the explicit `residence_allocation` timeline
    # step (see run_timeline below), not implicitly before the timeline.
    # The World starts with no distributor; the step attaches it when it runs.
    household_distributor = None

    # Create World object
    logger.info("")
    logger.info("Creating World object...")
    world = World(geography=geo, population=population, venues=venues, household_distributor=household_distributor)
    bp.bind(world=world)
    logger.info(world)

    try:
        run_timeline(world, config)
    except TimelineError:
        # Already logged by the failing step.
        sys.exit(1)

    # RELATIONSHIP PIPELINE - Build agent networks
    relationship_config = config.get("relationship_pipeline", {})

    if relationship_config.get("enabled", False):
        logger.info("")
        logger.info("=" * 60)
        logger.info("RELATIONSHIP PIPELINE")
        logger.info("=" * 60)

        relationship_configs = relationship_config.get("relationships", [])

        for rel_config in relationship_configs:
            config_path = pr.require_config_path(
                "Every `relationship_pipeline.relationships` entry", rel_config
            )

            logger.info("")
            logger.info(f"[RELATIONSHIP] {config_path}")

            with bp.stage(step_label("relationship", config_path), "relationship", detail=config_path):
                try:
                    builder = SocialNetworkBuilder.from_yaml(world, config_path)
                    builder.build_all()

                except Exception as e:
                    logger.error(f"Failed to build relationships from {config_path}: {e}")
                    logger.exception(e)
                    sys.exit(1)

    # ROMANTIC RELATIONSHIPS - Sexual orientation and partnerships
    romantic_config = config.get("romantic_relationships", {})

    if romantic_config.get("enabled", False):
        logger.info("")
        logger.info("=" * 60)
        logger.info("ROMANTIC RELATIONSHIPS")
        logger.info("=" * 60)

        config_path = pr.require_config_path(
            "`romantic_relationships.enabled: true`", romantic_config,
            " naming this scenario's romantic-relationships YAML",
        )

        with bp.stage("romantic_relationships", "romantic", detail=config_path):
            try:
                from may.relationships.romantic_relationships import RomanticDistributor
                distributor = RomanticDistributor(world, config_path)
                distributor.distribute_all()

            except Exception as e:
                logger.error(f"Failed to distribute romantic relationships: {e}")
                logger.exception(e)
                sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("World creation complete!")
    logger.info(f"Geography: {len(world.geography.get_all_units())} units")
    logger.info(f"Venues: {sum(len(d) for d in world.venues.venues_by_type_and_id.values())} venues across {len(venues.get_venue_types())} types")
    logger.info(f"Population: {len(world.population.get_all_people()):,} people")
    logger.info("=" * 60)

    # Export world to HDF5 for C++ simulation
    if serialisation.enabled:
        logger.info("")
        logger.info("Exporting world to HDF5...")
        with bp.stage("hdf5_export", "pipeline"):
            try:
                os.makedirs(serialisation.output_dir, exist_ok=True)
                export_path = os.path.join(
                    serialisation.output_dir, args.filename or serialisation.filename
                )
                world.export_to_hdf5(export_path, serialisation.schema)
            except Exception as e:
                logger.error(f"Failed to serialize world to HDF5: {e}")
                logger.exception(e)
                sys.exit(1)

    recorder = bp.active()
    if recorder is not None:
        recorder.finish()
        recorder.log_summary(logger)

    return world


if __name__ == "__main__":
    # Force a deterministic hash seed for reproducible runs. This re-execs the
    # interpreter, so it must stay in the CLI entry path. Doing it at import
    # time replaces the process whenever the module is imported (e.g. by tests).
    if os.environ.get('PYTHONHASHSEED') is None:
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable] + sys.argv)

    cli_args = parse_args()

    profiler = None
    if cli_args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    world = main(cli_args)

    if profiler is not None:
        profiler.disable()
        recorder = bp.active()
        if recorder is not None:
            profiler.dump_stats(recorder.stats_path)
