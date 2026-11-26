"""The distributor timeline step: one config stage for every `distributor_type`."""

from dataclasses import dataclass
from typing import Any, Dict

from may.preflight.config_reader import FAILED
from may.preflight.config_stage import ConfigStage

from .base_distributor import DistributorConfigError
from .multi_venue_distributor import MultiVenueDistributor
from .property_matching_distributor import PropertyMatchingDistributor
from .resident_linked_distributor import ResidentLinkedDistributor
from .route_distributor import RouteDistributor
from .social_contact_visit_distributor import SocialContactVisitDistributor
from .venue_distributor import VenueDistributor

DISTRIBUTOR_TYPES = {
    "single_venue": VenueDistributor,
    "multi_venue": MultiVenueDistributor,
    "social_contact_visit": SocialContactVisitDistributor,
    "resident_linked": ResidentLinkedDistributor,
    "property_matching": PropertyMatchingDistributor,
    "route": RouteDistributor,
}


@dataclass(frozen=True)
class ParsedDistributor:
    """The class `distributor_type` names, its checked config mapping and where the file was found."""
    distributor_class: type
    config_mapping: Dict[str, Any]
    config_path: str


class DistributorStep(ConfigStage):
    """A distributor YAML: `distributor_type` (default `single_venue`) picks the class whose
    `check_config` reads the rest, since it decides which of six schemas the file is written in."""

    display_name = "Distributor"
    config_error = DistributorConfigError
    config_is_file = True

    @classmethod
    def parse(cls, reader, context):
        distributor_type = reader.one_of("distributor_type", DISTRIBUTOR_TYPES, default="single_venue")
        if distributor_type is FAILED:
            return None
        distributor_class = DISTRIBUTOR_TYPES[distributor_type]
        distributor_class.check_config(reader, context)
        return ParsedDistributor(distributor_class, reader.value(), reader.found_path)

    @classmethod
    def build(cls, parsed, **live):
        """The chosen distributor from its parsed mapping; reads no file."""
        return parsed.distributor_class(config_dict=parsed.config_mapping, config_path=parsed.config_path)
