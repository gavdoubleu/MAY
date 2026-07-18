"""Tombstone semantics for flushed properties (see tmp/prd + ADR-0003).

A flushed key must stay visible to `in` (so membership checks don't silently
fall through, e.g. venue_distributor/filtering.py) but any read or write of
its value must fail loud with FlushedPropertyError.
"""

import pytest

from may.flushing import FlushedPropertyError, TombstonedProperties
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
