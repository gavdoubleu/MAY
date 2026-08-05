"""Comparing two exported worlds for equivalence.

Shared by the tests that assert a change to *when* data is written cannot
change *what* is written: two-stage export, timeline flush steps, and the
pending-flush bound.
"""

import h5py
import numpy as np

# Deliberate divergences from a baseline export: timestamps, and the
# inspection attr stamped on datasets written by flush_properties steps.
IGNORED_ATTRS = {'creation_timestamp', 'flushed_by_step'}


def assert_h5_equivalent(path_a, path_b):
    """Recursively compare two HDF5 files: same tree, datasets and attrs."""

    def compare_attrs(obj_a, obj_b, location):
        keys_a = set(obj_a.attrs.keys()) - IGNORED_ATTRS
        keys_b = set(obj_b.attrs.keys()) - IGNORED_ATTRS
        assert keys_a == keys_b, f"attr keys differ at {location}: {keys_a ^ keys_b}"
        for key in keys_a:
            value_a, value_b = obj_a.attrs[key], obj_b.attrs[key]
            assert np.array_equal(value_a, value_b), f"attr {key} differs at {location}"

    def compare_group(group_a, group_b, location):
        compare_attrs(group_a, group_b, location)
        keys_a = set(group_a.keys())
        keys_b = set(group_b.keys())
        assert keys_a == keys_b, f"children differ at {location}: {keys_a ^ keys_b}"
        for key in keys_a:
            child_a, child_b = group_a[key], group_b[key]
            child_location = f"{location}/{key}"
            assert type(child_a) is type(child_b), f"node type differs at {child_location}"
            if isinstance(child_a, h5py.Group):
                compare_group(child_a, child_b, child_location)
            else:
                compare_attrs(child_a, child_b, child_location)
                data_a, data_b = child_a[()], child_b[()]
                assert np.array_equal(data_a, data_b, equal_nan=(data_a.dtype.kind == 'f')), \
                    f"dataset differs at {child_location}"

    with h5py.File(path_a, 'r') as file_a, h5py.File(path_b, 'r') as file_b:
        compare_group(file_a, file_b, '')
