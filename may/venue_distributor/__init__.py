"""
Venue Distributor Module

This module provides a flexible, YAML-driven system for distributing people to venues.
The system reads configuration from YAML files in yaml/distributors/ and allocates
people to venues based on attributes, distance, and capacity constraints.

Classes:
    VenueDistributor: Main class for single-venue allocation
    MultiVenueDistributor: Class for multi-venue allocation (multiple venues per person)
    DistributorStep: The timeline step's config stage; `distributor_type` picks the class
"""

from .base_distributor import DistributorConfigError
from .venue_distributor import VenueDistributor
from .multi_venue_distributor import MultiVenueDistributor
from .social_contact_visit_distributor import SocialContactVisitDistributor
from .resident_linked_distributor import ResidentLinkedDistributor
from .property_matching_distributor import PropertyMatchingDistributor
from .route_distributor import RouteDistributor
from .distributor_step import DISTRIBUTOR_TYPES, DistributorStep, ParsedDistributor

__all__ = [
    "VenueDistributor",
    "MultiVenueDistributor",
    "SocialContactVisitDistributor",
    "ResidentLinkedDistributor",
    "PropertyMatchingDistributor",
    "RouteDistributor",
    "DistributorConfigError",
    "DistributorStep",
    "ParsedDistributor",
    "DISTRIBUTOR_TYPES",
]
