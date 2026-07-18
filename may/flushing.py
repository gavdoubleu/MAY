"""Tombstone support for properties flushed to the canonical HDF5 file.

Once a property has been written to disk by a `flush_properties` timeline
step, its in-memory values are freed. The key stays visible to `in` so
membership checks (e.g. may/venue_distributor/filtering.py) don't silently
fall through, but reading or writing the value fails loud: the value now
lives only on disk, and a post-flush write could never reach the export.
"""


class FlushedPropertyError(RuntimeError):
    """Read or write of a property already flushed to disk.

    Deliberately NOT a KeyError: existing `except KeyError` handlers must
    not swallow this — a flush-schedule mistake has to surface.
    """


class TombstonedProperties(dict):
    """A properties dict consulting a shared registry of flushed keys.

    The registry set is shared across every object of the flush target
    (e.g. all Persons), so flushing costs one set — not per-object state.
    """

    __slots__ = ('_flushed_keys',)

    def __init__(self, data, flushed_keys):
        super().__init__(data)
        self._flushed_keys = flushed_keys

    def _refuse_if_flushed(self, key):
        if key in self._flushed_keys:
            raise FlushedPropertyError(
                f"Property {key!r} was flushed to the canonical HDF5 file; "
                f"its in-memory value is gone. Reorder the timeline so this "
                f"access happens before the flush_properties step."
            )

    def __contains__(self, key):
        return key in self._flushed_keys or dict.__contains__(self, key)

    def __getitem__(self, key):
        self._refuse_if_flushed(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._refuse_if_flushed(key)
        return dict.get(self, key, default)

    def __setitem__(self, key, value):
        self._refuse_if_flushed(key)
        dict.__setitem__(self, key, value)

    def setdefault(self, key, default=None):
        self._refuse_if_flushed(key)
        return dict.setdefault(self, key, default)

    def pop(self, key, *args):
        self._refuse_if_flushed(key)
        return dict.pop(self, key, *args)

    def update(self, *args, **kwargs):
        # dict.update bypasses __setitem__ at C level; route through it so
        # flushed keys are refused here too.
        merged = dict(*args, **kwargs)
        for key in merged:
            self._refuse_if_flushed(key)
        dict.update(self, merged)


def tombstone_objects(objects, property_names, flushed_keys):
    """Free flushed values and install tombstones on every object.

    Adds `property_names` to the shared `flushed_keys` registry, drops the
    values from each object's properties dict, and swaps the dict for a
    TombstonedProperties bound to the registry (idempotent per object).
    """
    flushed_keys.update(property_names)
    for obj in objects:
        properties = obj.properties
        if not isinstance(properties, TombstonedProperties):
            properties = TombstonedProperties(properties, flushed_keys)
            obj.properties = properties
        for name in property_names:
            dict.pop(properties, name, None)
