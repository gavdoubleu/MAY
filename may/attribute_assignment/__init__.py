"""
Attribute Assignment System for MAY.

This module provides a generic, rule-based system for assigning attributes
(ethnicity, religion, language, etc.) to people AFTER they have been allocated
to households and venues.

Key components:
- assignment_config: Check and build attribute configs (a timeline step stage)
- data_sources: Load demographic data from CSV files
- strategies: Assignment strategy implementations
- assigner: Main orchestrator for attribute assignment

Usage:
    from may.attribute_assignment import assign_attributes

    stats = assign_attributes(
        venue_manager=venue_manager,
        config=attribute_config,  # an AttributeAssignmentConfig
        geo_units={'E00000001', 'E00000002'}  # Optional
    )
"""

from .assignment_config import AttributeAssignmentConfig, AttributeAssignmentError, AttributeConfigError
from .strategies import create_strategy
from .assigner import AttributeAssigner, assign_attributes

__all__ = [
    "AttributeAssignmentConfig",
    "create_strategy",
    "AttributeAssigner",
    "AttributeAssignmentError",
    "AttributeConfigError",
    "assign_attributes",
]
