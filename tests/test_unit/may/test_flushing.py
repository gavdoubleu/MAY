"""Tombstone semantics for flushed properties (see tmp/prd + ADR-0003).

A flushed key must stay visible to `in` (so membership checks don't silently
fall through, e.g. venue_distributor/filtering.py) but any read or write of
its value must fail loud with FlushedPropertyError.
"""

import pytest

from may.flushing import FlushedPropertyError, TombstonedProperties


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
