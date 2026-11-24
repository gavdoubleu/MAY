"""
Venue Distributor Module

This module provides a flexible, YAML-driven system for distributing people to venues.
The system reads configuration from YAML files in yaml/distributors/ and allocates
people to venues based on attributes, distance, and capacity constraints.

Classes:
    VenueDistributor: Main class for single-venue allocation
    MultiVenueDistributor: Class for multi-venue allocation (multiple venues per person)

Functions:
    distributor_from_yaml: Factory function to load the appropriate distributor from YAML
"""

from may.preflight.config_file import load_config_mapping
from may.preflight.report import PreflightReport

from .base_distributor import DistributorConfigError
from .venue_distributor import VenueDistributor
from .multi_venue_distributor import MultiVenueDistributor
from .social_contact_visit_distributor import SocialContactVisitDistributor
from .resident_linked_distributor import ResidentLinkedDistributor
from .property_matching_distributor import PropertyMatchingDistributor
from .route_distributor import RouteDistributor

DISTRIBUTOR_TYPES = {
    "single_venue": VenueDistributor,
    "multi_venue": MultiVenueDistributor,
    "social_contact_visit": SocialContactVisitDistributor,
    "resident_linked": ResidentLinkedDistributor,
    "property_matching": PropertyMatchingDistributor,
    "route": RouteDistributor,
}

__all__ = [
    "VenueDistributor",
    "MultiVenueDistributor",
    "SocialContactVisitDistributor",
    "ResidentLinkedDistributor",
    "PropertyMatchingDistributor",
    "RouteDistributor",
    "DistributorConfigError",
    "distributor_from_yaml",
    "distributor_preflight",
    "DISTRIBUTOR_TYPES",
]


def _distributor_class(config_path):
    """The class `distributor_type` names, or `None` with its problem in the report.

    `distributor_type` decides which of six config schemas the rest of the file is
    written in, so an unrecognised one is a mistake rather than an instruction."""
    report = PreflightReport()
    config = load_config_mapping(config_path, report)
    if config is None:
        return None, report
    distributor_type = config.get("distributor_type", "single_venue")
    if isinstance(distributor_type, (list, dict)) or distributor_type not in DISTRIBUTOR_TYPES:
        report.problem("distributor_type", (
            f"unknown distributor type {distributor_type!r}; expected one of {sorted(DISTRIBUTOR_TYPES)}"
        ))
        return None, report
    return DISTRIBUTOR_TYPES[distributor_type], report


def distributor_preflight(config_path, context):
    """Check a distributor YAML with the class its `distributor_type` names, building nothing."""
    distributor_class, report = _distributor_class(config_path)
    if distributor_class is None:
        return report
    return distributor_class.preflight(config_path, context)


def distributor_from_yaml(config_path, world_config=None):
    """Build the distributor its `distributor_type` names (`single_venue` when absent).

    Raises `DistributorConfigError` on the problems `distributor_preflight` reports."""
    distributor_class, report = _distributor_class(config_path)
    report.raise_if_problems(DistributorConfigError, f"Distributor config {config_path} cannot be used.")
    return distributor_class.from_config(config_path, world_config)
