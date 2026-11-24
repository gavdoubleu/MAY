"""
Configuration system for attribute assignment.

Simplified attribute assignment configuration:
- Roles are mapped to household subsets (Kids, Young Adults, Adults, Old Adults)
- Household structures use flexible pattern matching with actual/original conditions
- Assignment rules are organized by household structure type
- Cleaner, more user-friendly configuration format
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from may.residence.composition_pattern import CompositionPattern
from may.attribute_assignment.strategies import (
    raise_on_invalid_assignment,
    validate_assignment_config,
)
from may.preflight.config_file import load_config_mapping
from may.preflight.context import PreflightContext
from may.preflight.report import PreflightReport, not_list_message, not_mapping_message
from may.utils import path_resolver as pr
from may.utils.yaml_loader import load_yaml

logger = logging.getLogger("may.attribute_assignment.config")


@lru_cache(maxsize=4096)
def _pattern_matches_cached(actual: str, template: str) -> bool:
    """
    Cached pattern matching function.

    Check if an actual pattern matches a template pattern.

    Examples:
        actual="2 0 2 0", template=">=1 >=0 2 0" -> True
        actual="0 1 2 0", template=">=1 >=0 2 0" -> False
    """
    if not actual or not template:
        return False

    try:
        actual_counts = [int(x) for x in actual.split()]
    except ValueError:
        return False

    template_pattern = CompositionPattern.from_string(template)

    if len(actual_counts) != len(template_pattern.requirements):
        return False

    for i, actual_count in enumerate(actual_counts):
        operator, required_count = template_pattern.requirements[i]

        if operator == "exact":
            if actual_count != required_count:
                return False
        elif operator == "gte":
            if actual_count < required_count:
                return False
        elif operator == "lte":
            if actual_count > required_count:
                return False

    return True


@dataclass
class MatchingRule:
    """
    A rule for matching household patterns.

    Can match based on:
    - actual pattern only
    - original pattern only
    - both actual AND original patterns (conditional matching)
    """

    actual_patterns: List[str] = field(default_factory=list)
    original_patterns: List[str] = field(default_factory=list)
    description: str = ""

    def matches(self, household, verbose: bool = False) -> bool:
        """
        Check if a household matches this rule.

        Args:
            household: Venue object with original_pattern and actual_pattern properties
            verbose: If True, log matching details

        Returns:
            True if household matches this rule
        """
        original_pattern = household.properties.get("original_pattern", "")

        # Compute actual pattern from household members
        actual_pattern = self._compute_actual_pattern(household)

        if verbose:
            logger.debug(f"      Testing matching rule:")
            logger.debug(f"        Description: {self.description}")
            logger.debug(
                f"        Household: original='{original_pattern}', actual='{actual_pattern}'"
            )

        # If both actual and original are specified, BOTH must match
        if self.actual_patterns and self.original_patterns:
            actual_match = self._matches_any_pattern(
                actual_pattern, self.actual_patterns
            )
            original_match = original_pattern in self.original_patterns

            if verbose:
                logger.debug(f"        Actual match: {actual_match}")
                logger.debug(f"        Original match: {original_match}")

            return actual_match and original_match

        # If only actual patterns specified
        if self.actual_patterns:
            match = self._matches_any_pattern(actual_pattern, self.actual_patterns)
            if verbose:
                logger.debug(f"        Actual match: {match}")
            return match

        # If only original patterns specified
        if self.original_patterns:
            match = original_pattern in self.original_patterns
            if verbose:
                logger.debug(f"        Original match: {match}")
            return match

        # No patterns specified - always matches
        return True

    def _matches_any_pattern(
        self, actual_pattern: str, template_patterns: List[str]
    ) -> bool:
        """
        Check if actual pattern matches any of the template patterns.
        Uses CompositionPattern for flexible matching with >=, <=, etc.
        """
        for template in template_patterns:
            if self._pattern_matches(actual_pattern, template):
                return True
        return False

    def _compute_actual_pattern(self, household) -> str:
        """
        Compute the actual composition pattern from household members.

        Args:
            household: Venue object with members

        Returns:
            Pattern string like "2 0 2 0" (counts per category)
        """
        # Check if pattern is already cached on the household
        cached_pattern = household.properties.get("_cached_actual_pattern")
        if cached_pattern is not None:
            return cached_pattern

        # Get age categories from household properties
        age_categories = household.properties.get("_age_categories", [])
        if not age_categories:
            return ""

        # Build category name → index mapping
        category_indices = {cat.name: i for i, cat in enumerate(age_categories)}

        # Count members in each category
        counts = [0] * len(age_categories)

        members = household.get_all_members()
        for person in members:
            # Get person's household category from activity_map
            # UNIFIED STRUCTURE: activity_map['residence']['household'] = [subsets]
            if (
                "residence" in person.activity_map
                and "household" in person.activity_map["residence"]
                and person.activity_map["residence"]["household"]
            ):
                subset_name = person.activity_map["residence"]["household"][
                    0
                ].subset_name

                if subset_name in category_indices:
                    counts[category_indices[subset_name]] += 1

        # Return as space-separated string
        pattern = " ".join(str(c) for c in counts)

        # Cache the pattern on the household for future lookups
        household.properties["_cached_actual_pattern"] = pattern

        return pattern

    def _pattern_matches(self, actual: str, template: str) -> bool:
        """
        Check if an actual pattern matches a template pattern.

        Examples:
            actual="2 0 2 0", template=">=1 >=0 2 0" -> True
            actual="0 1 2 0", template=">=1 >=0 2 0" -> False
            actual="2 1 2 0", template="0 >=1 1 <=2" -> False (has kids)
            actual="0 1 1 1", template="0 >=1 1 <=2" -> True
        """
        # Use cached function for pattern matching
        return _pattern_matches_cached(actual, template)


@dataclass
class HouseholdStructure:
    """
    Household structure with flexible matching rules.
    """

    name: str
    description: str
    matching_rules: List[MatchingRule] = field(default_factory=list)

    def matches(self, household, verbose: bool = False) -> bool:
        """
        Check if household matches this structure.
        Returns True if ANY matching rule matches.
        """
        if verbose:
            logger.debug(f"    Testing structure '{self.name}':")
            logger.debug(f"      {len(self.matching_rules)} matching rule(s)")

        for rule in self.matching_rules:
            if rule.matches(household, verbose=verbose):
                if verbose:
                    logger.debug(f"    ✓ MATCHED structure '{self.name}'")
                return True

        if verbose:
            logger.debug(f"    ✗ No match for structure '{self.name}'")
        return False


@dataclass
class Role:
    """
    Role definition - maps to household subsets instead of conditions.
    """

    name: str
    description: str
    subsets: List[str]  # List of subset names this role applies to
    role_type: str = "general"  # primary, secondary, extra, or general

    def matches(self, person, verbose: bool = False) -> bool:
        """
        Check if person's subset matches this role.
        """
        # Get person's subset from residence allocation
        # UNIFIED STRUCTURE: activity_map['residence'][venue_type] = [subsets]
        if "residence" not in person.activity_map:
            return False

        if not person.activity_map["residence"]:
            return False

        # Get the residence subset from any residence type (household, care_home, etc.)
        residence_dict = person.activity_map["residence"]
        residence_subset = None

        for venue_type, subsets in residence_dict.items():
            if subsets and isinstance(subsets, list) and len(subsets) > 0:
                residence_subset = subsets[0]
                break

        if (
            residence_subset is None
        ):  # Check for None explicitly, not truthiness (Subset has __len__)
            return False

        person_subset = residence_subset.subset_name

        if verbose:
            logger.debug(
                f"        Testing role '{self.name}': person_subset='{person_subset}', role_subsets={self.subsets}"
            )

        return person_subset in self.subsets


@dataclass
class AssignmentRule:
    """
    Simplified assignment rule.
    """

    role: str  # Can also be list of roles (parsed from config)
    priority: int
    description: str
    assignment: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)  # Roles this rule depends on

    def applies_to_role(self, role_name: str) -> bool:
        """Check if this rule applies to a role."""
        if isinstance(self.role, list):
            return role_name in self.role
        return role_name == self.role


@dataclass
class StructureAssignmentRules:
    """
    Assignment rules for a specific household structure.
    """

    structure_name: str
    description: str
    rules: List[AssignmentRule] = field(default_factory=list)


def _extract_role_dependencies(assignment: Dict[str, Any]) -> List[str]:
    """
    Collect every role this assignment references and must therefore be assigned
    after.

    A strategy may read another role's already-assigned value three ways:
    `inherit_from` (copy it), `partner_role` (match it), or `exclude` (differ
    from it). Each referenced role becomes an ordering dependency so the topo-sort
    assigns it first whenever both roles are present in a household. Only declared
    references count — an absent reference is the author choosing not to relate
    the roles, not a dependency.
    """
    deps: List[str] = []

    inherit_from = assignment.get("inherit_from") or {}
    if isinstance(inherit_from, dict):
        # Forward inheritance uses 'roles' (list); reverse uses 'role' (string).
        if "roles" in inherit_from:
            deps.extend(inherit_from["roles"])
        elif "role" in inherit_from:
            deps.append(inherit_from["role"])

    partner_role = assignment.get("partner_role")
    if partner_role:
        deps.append(partner_role)

    # `exclude` lists live inside declarative logic `then` blocks; each entry is a
    # "role.attr" reference to a role this one must differ from.
    for entry in assignment.get("logic") or []:
        then = entry.get("then") if isinstance(entry, dict) else None
        if isinstance(then, dict):
            for ref in then.get("exclude") or []:
                deps.append(ref.split(".", 1)[0])

    return deps


def role_dependency_problems(rules) -> List[str]:
    """
    Unorderable cross-role references in one structure, as messages.

    `rules` is `(role, dependencies)` pairs, `role` a name or list of names. Every
    dependency must name a role defined in this structure (else it is a typo), and
    the role-level dependency graph must be acyclic (else no assignment order can
    satisfy it). A role defined here that simply has no member in a given household
    is fine: that is handled quietly at assignment time, not a structural error.
    """
    def role_names(role):
        return [name for name in (role if isinstance(role, list) else [role]) if name is not None]

    defined_roles = set()
    for role, _ in rules:
        defined_roles.update(role_names(role))

    problems = []
    edges = defaultdict(set)
    for role, dependencies in rules:
        for dependency in dependencies:
            if dependency not in defined_roles:
                problems.append(
                    f"rule for role {role!r} references undefined role {dependency!r}. "
                    f"A cross-role reference (inherit_from/partner_role/exclude) must "
                    f"name a role defined in this structure."
                )
                continue
            for name in role_names(role):
                if name != dependency:
                    edges[name].add(dependency)

    cycle = _dependency_cycle(defined_roles, edges)
    if cycle:
        problems.append(
            f"cross-role references form a cycle ({' -> '.join(cycle)}); no assignment "
            f"order can satisfy them."
        )
    return problems


def _dependency_cycle(roles: set, edges: Dict[str, set]) -> Optional[List[str]]:
    """One cycle in the role dependency graph, as a closed path, or `None`."""
    unvisited, in_progress, finished = 0, 1, 2
    state = {role: unvisited for role in roles}
    path: List[str] = []

    def visit(role: str) -> Optional[List[str]]:
        state[role] = in_progress
        path.append(role)
        for dependency in sorted(edges.get(role, ())):
            if state[dependency] == in_progress:
                return path[path.index(dependency):] + [dependency]
            if state[dependency] == unvisited:
                cycle = visit(dependency)
                if cycle:
                    return cycle
        path.pop()
        state[role] = finished
        return None

    for role in sorted(roles):
        if state[role] == unvisited:
            cycle = visit(role)
            if cycle:
                return cycle
    return None


class AttributeAssignmentError(Exception):
    """Raised when assignment can't complete on the given data/config. Mirrors
    PopulationError/VenueError/HouseholdError: every eligible person is assigned
    on complete data, or the build fails loud. A person left unassigned for any
    reason (strategy failure, unclassifiable residence, no matching role, no
    value) aborts the run."""


class AttributeAssignmentConfig:
    """
    Configuration loader for attribute assignment.

    Simplified configuration system:
    - Roles map to subsets
    - Structures use matching_rules
    - Assignment rules organized by structure
    """

    def __init__(self, config_path: Path):
        """Load configuration from YAML."""
        self.config_path = Path(pr.resolve(str(config_path)))

        self.raw_config = load_yaml(self.config_path)

        # Parse sections
        self.attributes = self._parse_attributes()
        # Primary produced attribute (single-output configs have exactly one).
        self.attribute_name = self.attributes[0]["name"]
        self.produced_attributes = [a["name"] for a in self.attributes]
        self.assignment_level = self._parse_assignment_level()
        self.residence_venue_types = self._parse_residence_venue_types()
        self.filters = self._parse_filters()
        self.required_attributes = self._parse_required_attributes()
        self.categories = self._parse_categories()
        self.roles = self._parse_roles()
        self.household_structures = self._parse_household_structures()
        self.data_sources = self._parse_data_sources()
        self.assignment_rules = self._parse_assignment_rules()
        self.venue_assignment_rules = self._parse_venue_assignment_rules()
        self.settings = self._parse_settings()

        # Cache valid roles per structure
        self._valid_roles_cache = {}

        # Cache category lookups
        self._category_lookup_cache = {}
        self._build_category_lookup_structures()

        logger.info(
            f"Loaded config for '{self.attribute_name}' from {self.config_path}"
        )
        logger.info(f"  Assignment level: {self.assignment_level}")
        if self.required_attributes:
            logger.info(
                f"  Required attributes: {list(self.required_attributes.keys())}"
            )
        if self.categories:
            logger.info(f"  Categories: {len(self.categories)}")
        logger.info(f"  Roles: {len(self.roles)}")
        logger.info(f"  Household structures: {len(self.household_structures)}")
        logger.info(f"  Assignment rules: {len(self.assignment_rules)}")

    def _parse_attributes(self) -> List[Dict[str, Any]]:
        """Parse the produced-attribute declarations.

        Produced attributes are always a top-level `attributes:` list; each entry
        carries a 'name'. A single-output config declares a one-entry list. Step
        config (assignment_level, residence_venue_types) lives under `step:`.
        A top-level singular `attribute:` block raises a clear error.
        """
        if "attribute" in self.raw_config:
            raise ValueError(
                "top-level 'attribute:' block is retired. Declare the "
                "produced attribute(s) as an 'attributes:' list and step config "
                "under 'step:'."
            )
        raw = self.raw_config.get("attributes")
        if not raw or not isinstance(raw, list):
            raise ValueError(
                "config needs an 'attributes:' list declaring the produced "
                "attribute(s), e.g. `attributes:\\n  - name: ethnicity`."
            )
        for entry in raw:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(
                    f"each 'attributes' entry needs a 'name', got {entry!r}"
                )
        return raw

    def _parse_assignment_level(self) -> str:
        """Parse assignment level: 'person' or 'person_by_residence'."""
        return self.raw_config.get("step", {}).get(
            "assignment_level", "person_by_residence"
        )

    def _parse_residence_venue_types(self) -> List[str]:
        """Residence venue types assigned by household structure (default ['household']).

        Other residence types fall through to venue_assignment_rules.
        """
        return self.raw_config.get("step", {}).get(
            "residence_venue_types", ["household"]
        )

    def _parse_filters(self) -> Dict[str, Any]:
        """Parse filters (e.g., activity-based filtering)."""
        return self.raw_config.get("filters", {})

    def _parse_required_attributes(self) -> Dict[str, Any]:
        """Parse required attributes (dependencies).

        Canonical form is a list of entries, each carrying a 'name':

            required_attributes:
              - name: ethnicity
                required: true

        Returned as a dict keyed by name for internal lookup. The mapping form
        (`name: {...}`) raises a clear error.
        """
        raw_attrs = self.raw_config.get("required_attributes", [])
        if not raw_attrs:
            return {}

        if not isinstance(raw_attrs, list):
            raise ValueError(
                f"required_attributes must be a list of entries each with a "
                f"'name', got {type(raw_attrs).__name__}. Convert the "
                f"mapping form `name:\\n  ...` to `- name: name\\n  ...`."
            )

        result = {}
        for attr in raw_attrs:
            if "name" not in attr:
                raise ValueError(
                    f"Required attribute entry missing 'name' field: {attr}"
                )
            name = attr["name"]
            result[name] = {k: v for k, v in attr.items() if k != "name"}
        return result

    def _parse_categories(self) -> List[Dict[str, Any]]:
        """Parse categories (e.g., age bands)."""
        return self.raw_config.get("categories", [])

    def _parse_roles(self) -> Dict[str, Role]:
        """Parse role definitions."""
        roles = {}
        roles_config = self.raw_config.get("roles", {})

        for role_name, role_data in roles_config.items():
            if not isinstance(role_data, dict):
                continue

            # Use explicit 'type' if provided, otherwise infer from name prefix.
            # The config builder writes the type AS the name prefix
            # (primary_/secondary_/extra_) and omits the explicit key, but
            # hand-authored configs may still use arbitrary names + explicit type.
            role_type = role_data.get("type")
            if not role_type:
                if role_name.startswith("primary_"):
                    role_type = "primary"
                elif role_name.startswith("secondary_"):
                    role_type = "secondary"
                elif role_name.startswith("extra_"):
                    role_type = "extra"
                else:
                    role_type = "general"

            roles[role_name] = Role(
                name=role_name,
                description=role_data.get("description", ""),
                subsets=role_data.get("subsets", []),
                role_type=role_type,
            )

        return roles

    def _parse_household_structures(self) -> Dict[str, HouseholdStructure]:
        """Parse household structure definitions."""
        structures = {}
        structures_config = self.raw_config.get("household_structures", {})

        for struct_name, struct_data in structures_config.items():
            if not isinstance(struct_data, dict):
                continue

            # Parse matching rules
            matching_rules = []
            for rule_data in struct_data.get("matching_rules", []):
                matching_rules.append(
                    MatchingRule(
                        actual_patterns=rule_data.get("actual", []),
                        original_patterns=rule_data.get("original", []),
                        description=rule_data.get("description", ""),
                    )
                )

            structures[struct_name] = HouseholdStructure(
                name=struct_name,
                description=struct_data.get("description", ""),
                matching_rules=matching_rules,
            )

        return structures

    def _parse_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Keep source settings in their validated YAML shape."""
        return self.raw_config.get("data_sources", {})

    def _parse_assignment_rules(self) -> Dict[str, StructureAssignmentRules]:
        """Parse structure-based assignment rules."""
        structure_rules = {}
        rules_config = self.raw_config.get("assignment_rules", {})

        for structure_name, struct_rules_data in rules_config.items():
            if not isinstance(struct_rules_data, dict):
                continue

            rules = []
            for i, rule_data in enumerate(struct_rules_data.get("rules", [])):
                assignment_data = rule_data.get("assignment", {})
                # Fail loudly on keys no strategy reads (dead config / typos).
                raise_on_invalid_assignment(
                    assignment_data,
                    where=f"{self.config_path.name}: assignment_rules."
                    f"{structure_name}.rules[{i}]",
                )

                # Ordering dependencies derived from every cross-role reference
                # (inherit_from/partner_role/exclude).
                dependencies = _extract_role_dependencies(assignment_data)

                rules.append(
                    AssignmentRule(
                        role=rule_data.get("role"),  # Can be string or list
                        priority=rule_data.get("priority", 999),
                        description=rule_data.get("description", ""),
                        assignment=assignment_data,
                        dependencies=list(set(dependencies)),
                    )
                )

            # Sort rules by priority
            rules.sort(key=lambda r: r.priority)

            # Ordering failures fail loud at load: a reference to an undefined role
            # is a typo, a cycle is genuinely unorderable.
            for message in role_dependency_problems(
                [(rule.role, rule.dependencies) for rule in rules]
            ):
                raise ValueError(
                    f"{self.config_path.name}: assignment_rules.{structure_name}: {message}"
                )

            structure_rules[structure_name] = StructureAssignmentRules(
                structure_name=structure_name,
                description=struct_rules_data.get("description", ""),
                rules=rules,
            )

        return structure_rules

    def _parse_venue_assignment_rules(self) -> List[Dict[str, Any]]:
        """Parse venue assignment rules."""
        rules = self.raw_config.get("venue_assignment_rules", [])
        for i, rule in enumerate(rules):
            raise_on_invalid_assignment(
                (rule or {}).get("assignment", {}),
                where=f"{self.config_path.name}: venue_assignment_rules[{i}]",
            )
        return rules

    def _parse_settings(self) -> Dict[str, Any]:
        """Parse settings."""
        return self.raw_config.get("settings", {})

    def get_household_structure(
        self, household, verbose: bool = False
    ) -> Optional[str]:
        """
        Classify household structure.
        Returns first matching structure.
        """
        # Check if structure is already cached (only when not verbose)
        if not verbose:
            cached_structure = household.properties.get("_cached_household_structure")
            if cached_structure is not None:
                return cached_structure

        if verbose:
            logger.debug(f"  Classifying household {household.id}:")

        for struct_name, structure in self.household_structures.items():
            if structure.matches(household, verbose=verbose):
                # Cache the result (only when not verbose to avoid caching debug runs)
                if not verbose:
                    household.properties["_cached_household_structure"] = struct_name
                return struct_name

        if verbose:
            logger.debug(f"  ✗ No structure matched")

        # Cache None result as well
        if not verbose:
            household.properties["_cached_household_structure"] = None

        return None

    def get_person_role(
        self,
        person,
        household_structure: str,
        assigned_roles: List[str],
        verbose: bool = False,
        person_category: str = None,
    ) -> Optional[str]:
        """
        Determine person's role based on their subset and household structure.

        Args:
            person: Person object
            household_structure: Name of household structure
            assigned_roles: List of roles already assigned in this household
            verbose: If True, log matching details
            person_category: Optional pre-calculated person category (subset name)

        Returns:
            Role name or None
        """
        if verbose:
            logger.debug(f"    Determining role for {person}:")

        # Get assignment rules for this structure
        if household_structure not in self.assignment_rules:
            if verbose:
                logger.debug(
                    f"      No assignment rules for structure '{household_structure}'"
                )
            return None

        struct_rules = self.assignment_rules[household_structure]

        # Get valid roles for this structure (cached to avoid rebuilding for every person)
        if household_structure not in self._valid_roles_cache:
            valid_roles_for_structure = set()
            for rule in struct_rules.rules:
                if isinstance(rule.role, list):
                    valid_roles_for_structure.update(rule.role)
                else:
                    valid_roles_for_structure.add(rule.role)
            self._valid_roles_cache[household_structure] = valid_roles_for_structure
        else:
            valid_roles_for_structure = self._valid_roles_cache[household_structure]

        # Try each role in order until we find a matching one
        for role_name, role in self.roles.items():
            # Skip roles that don't have rules for this structure
            if role_name not in valid_roles_for_structure:
                continue

            if verbose:
                logger.debug(f"      Testing role '{role_name}':")

            # Check if person's subset matches this role
            # Use pre-calculated category if valid
            matched = False
            if person_category and person_category != "unknown":
                if person_category in role.subsets:
                    matched = True
                elif verbose:
                    logger.debug(
                        f"        ✗ Category '{person_category}' not in role subsets {role.subsets}"
                    )
            else:
                # Otherwise use internal lookup
                if role.matches(person, verbose=verbose):
                    matched = True

            if not matched:
                continue

            # Check if this role has been assigned already
            role_count = assigned_roles.count(role_name)

            # Determine if we should assign this role based on count and explicit type
            if role.role_type == "primary" and role_count == 0:
                if verbose:
                    logger.debug(f"      ✓ Assigned role '{role_name}' (primary)")
                return role_name

            elif role.role_type == "secondary" and role_count == 0:
                # Check if a primary role for the same subset was already assigned
                has_primary = False
                for assigned_name in assigned_roles:
                    assigned_role = self.roles.get(assigned_name)
                    if assigned_role and assigned_role.role_type == "primary":
                        # Check if subsets overlap (e.g., both apply to 'Adults')
                        if set(assigned_role.subsets) & set(role.subsets):
                            has_primary = True
                            break

                if has_primary:
                    if verbose:
                        logger.debug(f"      ✓ Assigned role '{role_name}' (secondary)")
                    return role_name

            elif role.role_type == "extra":
                # Check if both primary and secondary for the same subset were assigned
                has_primary = False
                has_secondary = False
                for assigned_name in assigned_roles:
                    assigned_role = self.roles.get(assigned_name)
                    if assigned_role:
                        if assigned_role.role_type == "primary" and (
                            set(assigned_role.subsets) & set(role.subsets)
                        ):
                            has_primary = True
                        if assigned_role.role_type == "secondary" and (
                            set(assigned_role.subsets) & set(role.subsets)
                        ):
                            has_secondary = True

                if has_primary and has_secondary:
                    if verbose:
                        logger.debug(f"      ✓ Assigned role '{role_name}' (extra)")
                    return role_name

            elif role.role_type == "general":
                if verbose:
                    logger.debug(f"      ✓ Assigned role '{role_name}'")
                return role_name

        if verbose:
            logger.debug(f"      ✗ No role matched")
        return None

    def get_assignment_rule(
        self, household_structure: str, role: str, verbose: bool = False
    ) -> Optional[AssignmentRule]:
        """
        Get assignment rule for a role within a structure.
        """
        if household_structure not in self.assignment_rules:
            return None

        struct_rules = self.assignment_rules[household_structure]

        for rule in struct_rules.rules:
            if rule.applies_to_role(role):
                if verbose:
                    logger.debug(f"    ✓ Found rule for role '{role}'")
                return rule

        if verbose:
            logger.debug(f"    ✗ No rule for role '{role}'")
        return None

    def _build_category_lookup_structures(self):
        """
        Build lookup structures for categories.
        Called once during __init__ to avoid repeated iterations.
        """
        # Group categories by attribute name for faster filtering
        categories_by_attr = {}
        for category in self.categories:
            attr = category.get("attribute")
            if attr not in categories_by_attr:
                categories_by_attr[attr] = []
            categories_by_attr[attr].append(category)

        # For numerical categories (like age), sort by min value for binary search
        for attr, cats in list(categories_by_attr.items()):
            numerical_cats = [c for c in cats if c.get("type") == "numerical"]
            if numerical_cats:
                # Sort by min value
                numerical_cats.sort(key=lambda c: c["numerical"]["min"])
                categories_by_attr[attr + "_numerical"] = numerical_cats

        self._categories_by_attr = categories_by_attr

    def get_category_for_value(
        self, value: Any, attribute_name: str = "age"
    ) -> Optional[Dict[str, Any]]:
        """
        Find which category a value falls into.

        Args:
            value: The value to categorize (e.g., age=25)
            attribute_name: The attribute name to match against (default: "age")

        Returns:
            Category dict with 'csv_value' or None if no match
        """
        # Check cache first (87% hit rate based on profiling patterns)
        cache_key = (attribute_name, value)
        if cache_key in self._category_lookup_cache:
            return self._category_lookup_cache[cache_key]

        result = None

        # Use pre-filtered categories
        numerical_cats = self._categories_by_attr.get(attribute_name + "_numerical", [])
        if numerical_cats and isinstance(value, (int, float)):
            # For numerical, iterate through sorted categories (typically just 4-5)
            for category in numerical_cats:
                min_val = category["numerical"]["min"]
                max_val = category["numerical"].get("max")

                if max_val is None:
                    # No upper limit
                    if value >= min_val:
                        result = category
                        break
                elif min_val <= value <= max_val:
                    result = category
                    break
        else:
            # For categorical, check all categories for this attribute
            cats = self._categories_by_attr.get(attribute_name, [])
            for category in cats:
                if category.get("type") == "categorical":
                    allowed = category.get("categorical", {}).get("allowed_values", [])
                    if value in allowed:
                        result = category
                        break

        # Cache the result
        self._category_lookup_cache[cache_key] = result
        return result

    def get_person_assignment_rule(self) -> Optional[AssignmentRule]:
        """
        Get assignment rule for person-level assignment.

        Returns:
            First assignment rule from 'person' structure or None
        """
        if "person" not in self.assignment_rules:
            return None

        person_rules = self.assignment_rules["person"]
        if not person_rules.rules:
            return None

        return person_rules.rules[0]

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AttributeAssignmentConfig":
        """Load configuration from YAML file."""
        return cls(config_path)

    @classmethod
    def parse_config(cls, config_path, context=None):
        """Parse an attribute YAML into its mapping, or `None` if any problem.

        Data files are checked to exist, never opened. `context`, when given, holds
        the geo levels known to this step."""
        report = PreflightReport()
        config = load_config_mapping(config_path, report)
        if config is None:
            return None, report
        _check_config(config, context or PreflightContext(), report)
        if report.has_problems:
            return None, report
        return config, report

    @classmethod
    def preflight(cls, config_path, context):
        """Check an attribute YAML without loading its data."""
        _, report = cls.parse_config(config_path, context)
        return report

    @classmethod
    def from_config(cls, config_path, world_config=None):
        """Build from an attribute YAML; `AttributeAssignmentError` on any problem."""
        _, report = cls.parse_config(config_path)
        report.raise_if_problems(
            AttributeAssignmentError, f"Attribute config {config_path} cannot be used."
        )
        return cls(config_path)

    def assign_attributes(self, world):
        """Assign this config's attributes to everyone in `world`."""
        world.assign_attributes(str(self.config_path))


ASSIGNMENT_LEVELS = ("person", "person_by_residence")
DATA_SOURCE_TYPES = ("csv_lookup", "constant")
# Kept in step with `DataSourceManager._CSV_FORMATS`, which owns the format classes.
CSV_FORMATS = (
    "geo_distribution", "diversity", "pair", "multi_key", "origin_destination_matrix", "gu_sampler",
)


def _check_config(config, context, report):
    """Every check an attribute YAML gets before its data is loaded."""
    _check_attributes(config, report)
    step = config.get("step")
    if step is not None and not isinstance(step, dict):
        report.problem("step", not_mapping_message(step))
    elif step:
        level = step.get("assignment_level", "person_by_residence")
        if level not in ASSIGNMENT_LEVELS:
            report.problem("step.assignment_level", (
                f"unknown value {level!r}; expected one of {list(ASSIGNMENT_LEVELS)}"
            ))
    _named_entries("required_attributes", config.get("required_attributes"), report)
    source_names = _check_data_sources(config.get("data_sources"), context, report)
    _check_assignment_rules(config.get("assignment_rules"), source_names, report)
    for position, rule in _mappings("venue_assignment_rules", config.get("venue_assignment_rules"), report):
        _check_assignment(
            f"venue_assignment_rules[{position}].assignment", rule.get("assignment", {}), source_names, report
        )


def _check_attributes(config, report):
    if "attribute" in config:
        report.problem("attribute", (
            "the top-level 'attribute:' block is retired. Declare the produced attribute(s) "
            "as an 'attributes:' list and step config under 'step:'."
        ))
    attributes = config.get("attributes")
    if not attributes or not isinstance(attributes, list):
        report.problem("attributes", (
            "needs a list declaring the produced attribute(s), e.g. "
            "`attributes:\\n  - name: ethnicity`"
        ))
        return
    _named_entries("attributes", attributes, report)


def _mappings(key, items, report):
    """`(position, item)` for each mapping in a list, counted from 1; anything else is a problem."""
    if items is None:
        return []
    if not isinstance(items, list):
        report.problem(key, not_list_message(items))
        return []
    mappings = []
    for position, item in enumerate(items, start=1):
        if isinstance(item, dict):
            mappings.append((position, item))
        else:
            report.problem(f"{key}[{position}]", not_mapping_message(item))
    return mappings


def _named_entries(key, items, report):
    """`_mappings`, each of which must carry a `name`."""
    entries = _mappings(key, items, report)
    for position, entry in entries:
        if not entry.get("name"):
            report.problem(f"{key}[{position}].name", "not set")
    return entries


def _section(key, section, report):
    """The mapping under `key`, `{}` when absent; a non-mapping is a problem and reads as `{}`."""
    if section is None:
        return {}
    if not isinstance(section, dict):
        report.problem(key, not_mapping_message(section))
        return {}
    return section


def _check_data_sources(data_sources, context, report):
    """Each source's type, format and files; the declared source names (`None` if unreadable)."""
    if data_sources is not None and not isinstance(data_sources, dict):
        report.problem("data_sources", not_mapping_message(data_sources))
        return None
    data_sources = data_sources or {}
    for name, source in data_sources.items():
        key = f"data_sources.{name}"
        if not isinstance(source, dict):
            report.problem(key, not_mapping_message(source))
            continue
        source_type = source.get("type", "csv_lookup")
        if source_type not in DATA_SOURCE_TYPES:
            report.problem(f"{key}.type", (
                f"unknown value {source_type!r}; expected one of {list(DATA_SOURCE_TYPES)}"
            ))
            continue
        if source_type == "constant":
            continue
        data_format = source.get("format")
        if data_format not in CSV_FORMATS:
            report.problem(f"{key}.format", (
                f"needs an explicit format, one of {sorted(CSV_FORMATS)}; got {data_format!r}"
            ))
        for position, file_config in _mappings(f"{key}.files", source.get("files"), report):
            _check_data_file(f"{key}.files[{position}]", file_config, context, report)
    return set(data_sources)


def _check_data_file(key, file_config, context, report):
    """A data file exists (not opened) and the geo levels it names are known."""
    path = file_config.get("path")
    if not path:
        report.problem(f"{key}.path", "not set")
    elif not Path(pr.resolve(str(path))).exists():
        report.problem(f"{key}.path", f"data file not found: {pr.resolve(str(path))}")
    level_keys = [("destination_level", file_config.get("destination_level"))]
    unit_column = file_config.get("geographical_unit_column")
    if isinstance(unit_column, dict):
        level_keys.append(("geographical_unit_column.level", unit_column.get("level")))
    key_columns = file_config.get("key_columns")
    if isinstance(key_columns, dict):
        for column, column_config in key_columns.items():
            if isinstance(column_config, dict):
                level_keys.append((f"key_columns.{column}.level", column_config.get("level")))
    for level_key, level in level_keys:
        _check_geo_level(f"{key}.{level_key}", level, context, report)


def _check_geo_level(key, geo_level, context, report):
    """A geo level, when set, must be one the geography defines (skipped while unknown)."""
    if geo_level is None or context.geo_levels is None:
        return
    if geo_level not in context.geo_levels:
        report.problem(key, f"unknown geo level {geo_level!r}; expected one of {sorted(context.geo_levels)}")


def _check_assignment_rules(assignment_rules, source_names, report):
    """Each structure's rule assignments, then whether its cross-role references can be ordered."""
    for structure_name, structure in _section("assignment_rules", assignment_rules, report).items():
        structure_key = f"assignment_rules.{structure_name}"
        if not isinstance(structure, dict):
            report.problem(structure_key, not_mapping_message(structure))
            continue
        problems_before = len(report.problems)
        rules = _mappings(f"{structure_key}.rules", structure.get("rules", []), report)
        for position, rule in rules:
            _check_assignment(
                f"{structure_key}.rules[{position}].assignment", rule.get("assignment", {}), source_names, report
            )
            _check_role_references(f"{structure_key}.rules[{position}]", rule, report)
        # A malformed assignment has no dependable references, so ordering is not
        # checked on top of it.
        if len(report.problems) > problems_before:
            continue
        role_dependencies = [
            (rule.get("role"), set(_extract_role_dependencies(rule.get("assignment", {}))))
            for _, rule in rules
        ]
        for message in role_dependency_problems(role_dependencies):
            report.problem(structure_key, message)


def _check_role_name(key, name, report):
    """A role reference must be a name; a list or mapping cannot be ordered or looked up."""
    if name is not None and not isinstance(name, str):
        report.problem(key, f"must be a role name; got {name!r}")


def _check_role_references(rule_key, rule, report):
    """The rule's `role` and every role its assignment reads are names (or a list of names)."""
    role = rule.get("role")
    if isinstance(role, list):
        for position, name in enumerate(role, start=1):
            _check_role_name(f"{rule_key}.role[{position}]", name, report)
    else:
        _check_role_name(f"{rule_key}.role", role, report)
    assignment = rule.get("assignment")
    if not isinstance(assignment, dict):
        return
    key = f"{rule_key}.assignment"
    _check_role_name(f"{key}.partner_role", assignment.get("partner_role"), report)
    inherit_from = assignment.get("inherit_from")
    if isinstance(inherit_from, dict):
        _check_role_name(f"{key}.inherit_from.role", inherit_from.get("role"), report)
        roles = inherit_from.get("roles")
        if isinstance(roles, list):
            for position, name in enumerate(roles, start=1):
                _check_role_name(f"{key}.inherit_from.roles[{position}]", name, report)
        elif "roles" in inherit_from:
            report.problem(f"{key}.inherit_from.roles", f"must be a list of role names; got {roles!r}")
    logic = assignment.get("logic")
    for position, entry in enumerate(logic if isinstance(logic, list) else [], start=1):
        then = entry.get("then") if isinstance(entry, dict) else None
        exclude = then.get("exclude") if isinstance(then, dict) else None
        if isinstance(exclude, list):
            for exclude_position, reference in enumerate(exclude, start=1):
                if not isinstance(reference, str):
                    report.problem(f"{key}.logic[{position}].then.exclude[{exclude_position}]", (
                        f"must be a 'role.attribute' reference; got {reference!r}"
                    ))
        elif exclude is not None:
            report.problem(f"{key}.logic[{position}].then.exclude", (
                f"must be a list of 'role.attribute' references; got {exclude!r}"
            ))


def _check_assignment(key, assignment, source_names, report):
    """One assignment block, and that each `data_source` it names is declared."""
    problems_before = len(report.problems)
    validate_assignment_config(assignment, report, key)
    if len(report.problems) > problems_before or source_names is None:
        return
    named_sources = [(f"{key}.data_source", assignment.get("data_source"))]
    for position, entry in enumerate(assignment.get("logic") or [], start=1):
        then = entry.get("then")
        if isinstance(then, dict):
            named_sources.append((f"{key}.logic[{position}].then.data_source", then.get("data_source")))
    for source_key, source_name in named_sources:
        if source_name is not None and not isinstance(source_name, str):
            report.problem(source_key, f"must be a data source name; got {source_name!r}")
        elif source_name is not None and source_name not in source_names:
            report.problem(source_key, (
                f"unknown data source {source_name!r}; declared: {sorted(source_names)}"
            ))
