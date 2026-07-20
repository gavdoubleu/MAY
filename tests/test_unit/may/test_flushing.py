"""Tombstone semantics for flushed properties (see tmp/prd + ADR-0003).

A flushed key must stay visible to `in` (so membership checks don't silently
fall through, e.g. venue_distributor/filtering.py) but any read or write of
its value must fail loud with FlushedPropertyError.
"""

import pytest

from may.flushing import (
    FlushedPropertyError,
    GuttedMemberMetadata,
    GuttedMembers,
    TombstonedProperties,
    gut_subset,
)
from may.venue_distributor.base_distributor import BaseDistributor


@pytest.fixture
def tombstoned():
    flushed_keys = {"ethnicity"}
    return TombstonedProperties({"work_mode": "cycle"}, flushed_keys)


class TestTombstonedProperties:

    def test_flushed_key_membership_is_true(self, tombstoned):
        assert "ethnicity" in tombstoned

    def test_flushed_key_getitem_raises(self, tombstoned):
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            tombstoned["ethnicity"]

    def test_flushed_key_get_raises_instead_of_default(self, tombstoned):
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            tombstoned.get("ethnicity", "fallback")

    def test_flushed_key_write_raises(self, tombstoned):
        """Writing after flush would be silent data loss — final export skips
        flushed columns, so the new value could never reach disk."""
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            tombstoned["ethnicity"] = "new_value"

    def test_unflushed_keys_behave_like_a_dict(self, tombstoned):
        assert tombstoned["work_mode"] == "cycle"
        assert tombstoned.get("missing") is None
        assert tombstoned.get("missing", 7) == 7
        assert "missing" not in tombstoned
        tombstoned["age_band"] = "20-24"
        assert tombstoned["age_band"] == "20-24"

    def test_error_is_not_a_keyerror(self):
        """Must not be swallowed by existing `except KeyError` handlers."""
        assert not issubclass(FlushedPropertyError, KeyError)

    def test_flushed_key_update_raises(self, tombstoned):
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            tombstoned.update({"ethnicity": "new_value"})
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            tombstoned.update(ethnicity="new_value")

    def test_registry_is_shared_not_copied(self):
        flushed_keys = set()
        properties = TombstonedProperties({}, flushed_keys)
        assert "ethnicity" not in properties
        flushed_keys.add("ethnicity")
        assert "ethnicity" in properties
        with pytest.raises(FlushedPropertyError):
            properties["ethnicity"]

    def test_keys_raises_when_any_key_flushed(self, tombstoned):
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            tombstoned.keys()

    def test_items_raises_when_any_key_flushed(self, tombstoned):
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            tombstoned.items()

    def test_values_raises_when_any_key_flushed(self, tombstoned):
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            tombstoned.values()

    def test_iter_raises_when_any_key_flushed(self, tombstoned):
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            iter(tombstoned)

    def test_len_raises_when_any_key_flushed(self, tombstoned):
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            len(tombstoned)

    def test_dict_copy_raises_when_any_key_flushed(self, tombstoned):
        """dict(x) routes through keys()/__getitem__ for non-exact-dict
        subclasses, so it inherits the same fail-loud guarantee for free."""
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            dict(tombstoned)

    def test_enumeration_raises_even_for_object_that_never_held_flushed_key(self):
        """Cohort-wide registry: raises regardless of whether this specific
        object ever had the flushed property, mirroring `in`."""
        flushed_keys = {"ethnicity"}
        properties = TombstonedProperties({"work_mode": "cycle"}, flushed_keys)
        assert "ethnicity" not in dict.keys(properties)
        with pytest.raises(FlushedPropertyError, match="ethnicity"):
            properties.keys()

    def test_enumeration_works_normally_before_any_flush(self):
        flushed_keys = set()
        properties = TombstonedProperties({"work_mode": "cycle"}, flushed_keys)
        assert list(properties.keys()) == ["work_mode"]
        assert list(properties.items()) == [("work_mode", "cycle")]
        assert list(properties.values()) == ["cycle"]
        assert list(properties) == ["work_mode"]
        assert len(properties) == 1

    def test_single_key_access_unaffected_by_other_flushed_keys(self, tombstoned):
        """Direct access to a still-live key must not raise, even though
        `ethnicity` has been flushed for this cohort."""
        assert tombstoned["work_mode"] == "cycle"
        assert tombstoned.get("work_mode") == "cycle"


class _NestedResidence:
    """Minimal stand-in exposing a `properties` attribute for traversal."""

    def __init__(self, properties):
        self.properties = properties


class TestTombstoneTraversalInDistributor:
    """`TombstonedProperties` is a dict subclass; distributor path traversal
    must treat it as a dict (isinstance, not `type(...) is dict`) or unflushed
    keys silently become invisible to filters — see the university-distributor
    regression where a whole cohort selected 0 people after an unrelated flush.
    """

    def test_unflushed_key_readable_through_tombstoned_properties(self):
        flushed_keys = {"ethnicity"}
        properties = TombstonedProperties(
            {"original_pattern": "pattern_a"}, flushed_keys
        )
        residence = _NestedResidence(properties)
        traverse = BaseDistributor._get_nested_value_with_dict_support

        value = traverse(None, residence, "properties.original_pattern")

        assert value == "pattern_a"


class TestGuttedMembers:

    def test_len_returns_cached_count(self):
        gutted = GuttedMembers(3)

        assert len(gutted) == 3

    def test_iteration_raises(self):
        gutted = GuttedMembers(3)

        with pytest.raises(FlushedPropertyError):
            iter(gutted)

    def test_membership_check_raises(self):
        gutted = GuttedMembers(3)

        with pytest.raises(FlushedPropertyError):
            "anything" in gutted

    def test_add_raises(self):
        gutted = GuttedMembers(3)

        with pytest.raises(FlushedPropertyError):
            gutted.add("person")

    def test_remove_raises(self):
        gutted = GuttedMembers(3)

        with pytest.raises(FlushedPropertyError):
            gutted.remove("person")

    def test_uncovered_set_methods_raise_curated_error(self):
        """clear/update/pop/... fall through to __getattr__, which must
        raise FlushedPropertyError rather than a bare AttributeError."""
        gutted = GuttedMembers(3)

        for method_name in ("clear", "update", "pop", "union", "difference_update"):
            with pytest.raises(FlushedPropertyError):
                getattr(gutted, method_name)

    def test_in_place_union_raises(self):
        gutted = GuttedMembers(3)

        with pytest.raises(FlushedPropertyError):
            gutted |= {"person"}

    def test_dunder_probe_keeps_attributeerror_semantics(self):
        """Copy/pickle protocol lookups must see AttributeError, not the
        curated error, so generic machinery isn't broken by the sentinel."""
        gutted = GuttedMembers(3)

        with pytest.raises(AttributeError):
            gutted.__deepcopy__

    def test_len_still_answers_from_cached_count(self):
        assert len(GuttedMembers(7)) == 7


class TestGuttedMemberMetadata:

    def test_bool_raises(self):
        gutted = GuttedMemberMetadata()

        with pytest.raises(FlushedPropertyError):
            bool(gutted)

    def test_len_raises(self):
        gutted = GuttedMemberMetadata()

        with pytest.raises(FlushedPropertyError):
            len(gutted)

    def test_getitem_raises(self):
        gutted = GuttedMemberMetadata()

        with pytest.raises(FlushedPropertyError):
            gutted["field"]

    def test_iteration_raises(self):
        gutted = GuttedMemberMetadata()

        with pytest.raises(FlushedPropertyError):
            iter(gutted)

    def test_membership_check_raises(self):
        gutted = GuttedMemberMetadata()

        with pytest.raises(FlushedPropertyError):
            "field" in gutted


class _StubSubset:

    def __init__(self):
        self.members = {"person_a", "person_b"}
        self.member_metadata = {"height": {"person_a": 1.8}}


class TestGutSubset:

    def test_replaces_members_and_metadata_with_sentinels(self):
        subset = _StubSubset()

        gut_subset(subset, member_count=2)

        assert isinstance(subset.members, GuttedMembers)
        assert len(subset.members) == 2
        assert isinstance(subset.member_metadata, GuttedMemberMetadata)
