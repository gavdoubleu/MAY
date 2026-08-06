"""RouteLegsIndex — memory-bounded, columnar route-legs lookup.

Replaces loading the full national legs CSV into a pandas DataFrame plus a
per-row dict index (see tmp/prd/route-legs-index-memory-reduction.md). Reads
the CSV in chunks, filters each chunk to the requested mode class, and
converts filtered rows into columnar buffers (numpy arrays; string fields as
int32 codes with decode tables) immediately per chunk — the full filtered
table is never materialised as one pandas object. A single `pd.concat`
equivalent would transiently double-hold the filtered set as a pandas
object (~4.2 GiB estimated for bus); concatenating small numpy int arrays
at the end does not have that cost (a handful of int32 columns, no string
overhead).

Lookup keying: bus commutes are overwhelmingly single-leg journeys, so
(origin, dest, mode_class) is close to a 1:1 key over rows (~5.81M distinct
keys for 5.81M bus rows). A plain Python dict mapping
(str, str, str) -> (start, stop) at that cardinality costs ~1+ GiB in
CPython object overhead alone — reintroducing the same class of problem
this module exists to remove. Instead, origin/dest/mode_class are encoded
through the same small int32 code tables used for line/board/alight (MGU
vocabulary is ~thousands, not millions), packed into a single int64 per
row, and looked up via a sorted array + binary search — no Python dict
sized by row count anywhere in the index.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from may.common.code_table import CodeTable

logger = logging.getLogger("route_legs_index")

REQUIRED_COLUMNS = {"origin_mgu", "dest_mgu", "mode_class", "leg_idx", "line_id",
                    "board_mgu", "alight_mgu"}

STRING_FIELDS = ("line_id", "board_mgu", "alight_mgu")
KEY_FIELDS = ("origin_mgu", "dest_mgu", "mode_class")

# Bit widths for packing (origin_code, dest_code, mode_class_code) into one
# int64 lookup key: origin_code << (DEST_BITS + MODE_BITS) | dest_code <<
# MODE_BITS | mode_class_code. 2**32 for dest and 2**24 (implicit, whatever
# remains) for origin comfortably exceed England's MGU count (~7,200 MSOAs);
# 2**8 for mode_class covers far more than the handful of real classes.
# Total width (32 + 8 = 40 low bits, origin above that) stays well inside
# int64's 63 usable positive bits.
_DEST_BITS = 32
_MODE_BITS = 8
_ORIGIN_SHIFT = _DEST_BITS + _MODE_BITS


def _pack_key(origin_code: np.ndarray, dest_code: np.ndarray, mode_class_code: np.ndarray) -> np.ndarray:
    return ((origin_code.astype(np.int64) << _ORIGIN_SHIFT)
            | (dest_code.astype(np.int64) << _MODE_BITS)
            | mode_class_code.astype(np.int64))


class RouteLegsIndex:
    """(origin, dest, mode_class) -> legs, built from a chunked, mode-filtered
    read of the national legs CSV. See module docstring for the lookup
    design (sorted packed-int keys, not a per-row Python dict)."""

    def __init__(self, legs_table_path: str, class_filter: Optional[str] = None,
                 class_map: Optional[Dict[str, str]] = None,
                 leg_metadata: Optional[Dict[str, str]] = None,
                 chunk_size: int = 500_000):
        self.legs_table_path = legs_table_path
        self.class_filter = class_filter
        self.class_map = class_map or {}
        self.leg_metadata = leg_metadata or {}
        self.chunk_size = chunk_size

        # Columnar storage, populated by _load(). Empty if the file is
        # absent or holds no matching rows.
        self._leg_idx = np.empty(0, dtype=np.int64)
        self._meta_arrays: Dict[str, np.ndarray] = {}
        self._string_codes: Dict[str, np.ndarray] = {name: np.empty(0, dtype=np.int32)
                                                       for name in STRING_FIELDS}
        self._code_tables: Dict[str, CodeTable] = {name: CodeTable() for name in STRING_FIELDS}
        self._key_code_tables: Dict[str, CodeTable] = {name: CodeTable() for name in KEY_FIELDS}
        # Sorted (ascending) packed keys, one per distinct (origin, dest,
        # mode_class); group_starts/group_stops are aligned index-for-index.
        self._sorted_group_keys = np.empty(0, dtype=np.int64)
        self._group_starts = np.empty(0, dtype=np.int64)
        self._group_stops = np.empty(0, dtype=np.int64)

        self._load()

    def _load(self) -> None:
        legs_path = Path(self.legs_table_path)
        if not legs_path.exists():
            logger.warning(
                f"Legs table not found: {legs_path}. "
                f"All eligible people will be treated as route misses."
            )
            return

        mode_classes_keep = None
        if self.class_filter is not None:
            mapped = self.class_map.get(self.class_filter, self.class_filter)
            mode_classes_keep = {mapped}

        meta_cols = list(self.leg_metadata.values())
        header = pd.read_csv(legs_path, nrows=0)
        missing = REQUIRED_COLUMNS - set(header.columns)
        if missing:
            raise ValueError(f"legs table {legs_path} missing columns: {missing}")
        meta_missing = [c for c in meta_cols if c not in header.columns]
        if meta_missing:
            raise ValueError(
                f"legs table missing metadata columns referenced by config: {meta_missing}"
            )

        logger.info(f"Loading legs table: {legs_path}")

        # Per-chunk numpy buffers, concatenated once at the end. Small
        # (int32/int64 columns only, no string payload) so the final
        # concatenation is cheap, unlike a pandas-level concat.
        leg_idx_chunks: List[np.ndarray] = []
        meta_chunks: Dict[str, List[np.ndarray]] = {c: [] for c in meta_cols}
        string_code_chunks: Dict[str, List[np.ndarray]] = {name: [] for name in STRING_FIELDS}
        packed_key_chunks: List[np.ndarray] = []
        n_legs = 0

        for chunk in pd.read_csv(legs_path, chunksize=self.chunk_size):
            if mode_classes_keep is not None:
                chunk = chunk[chunk["mode_class"].isin(mode_classes_keep)]
            if chunk.empty:
                continue

            leg_idx_chunks.append(chunk["leg_idx"].to_numpy(dtype=np.int64))
            for c in meta_cols:
                meta_chunks[c].append(chunk[c].to_numpy())
            for name in STRING_FIELDS:
                string_code_chunks[name].append(self._code_tables[name].encode_series(chunk[name]))

            origin_codes = self._key_code_tables["origin_mgu"].encode_series(chunk["origin_mgu"])
            dest_codes = self._key_code_tables["dest_mgu"].encode_series(chunk["dest_mgu"])
            mode_class_codes = self._key_code_tables["mode_class"].encode_series(chunk["mode_class"])
            packed_key_chunks.append(_pack_key(origin_codes, dest_codes, mode_class_codes))

            n_legs += len(chunk)

        if n_legs == 0:
            logger.info("  Indexed 0 legs across 0 (O,D,class) routes")
            return

        self._leg_idx = np.concatenate(leg_idx_chunks)
        self._meta_arrays = {c: np.concatenate(meta_chunks[c]) for c in meta_cols}
        for name in STRING_FIELDS:
            self._string_codes[name] = np.concatenate(string_code_chunks[name])
        packed_keys = np.concatenate(packed_key_chunks)

        # Sort rows by packed key so same-key rows are contiguous. No
        # within-key ordering is required (PRD requirement 4), so a plain
        # (non-stable) sort is fine.
        order = np.argsort(packed_keys)
        self._leg_idx = self._leg_idx[order]
        self._meta_arrays = {c: arr[order] for c, arr in self._meta_arrays.items()}
        for name in STRING_FIELDS:
            self._string_codes[name] = self._string_codes[name][order]
        sorted_keys = packed_keys[order]

        boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
        self._group_starts = np.concatenate(([0], boundaries))
        self._group_stops = np.concatenate((boundaries, [len(sorted_keys)]))
        self._sorted_group_keys = sorted_keys[self._group_starts]

        logger.info(
            f"  Indexed {n_legs:,} legs across {len(self._sorted_group_keys):,} (O,class) routes"
        )

    def served_units(self) -> Set[str]:
        """Every MGU this index routes from or to.

        Callers use it to decide which units a rider can reach a stop in, so it
        must name exactly the units the surviving legs mention. It does: the
        mode filter is applied before the key columns are encoded, and no row
        is dropped afterwards, so the key code tables hold that set precisely.
        Sized by vocabulary, not by legs.
        """
        return (set(self._key_code_tables["origin_mgu"].decode)
                | set(self._key_code_tables["dest_mgu"].decode))

    def get(self, origin: str, dest: str, mode_class: str) -> Optional[List[Dict[str, Any]]]:
        origin_code = self._key_code_tables["origin_mgu"].code_of(origin)
        dest_code = self._key_code_tables["dest_mgu"].code_of(dest)
        mode_class_code = self._key_code_tables["mode_class"].code_of(mode_class)
        if origin_code is None or dest_code is None or mode_class_code is None:
            return None

        query_key = (origin_code << _ORIGIN_SHIFT) | (dest_code << _MODE_BITS) | mode_class_code
        position = np.searchsorted(self._sorted_group_keys, query_key)
        if position >= len(self._sorted_group_keys) or self._sorted_group_keys[position] != query_key:
            return None

        start = int(self._group_starts[position])
        stop = int(self._group_stops[position])
        legs = []
        for i in range(start, stop):
            leg = {
                "leg_idx": int(self._leg_idx[i]),
                "line_id": self._code_tables["line_id"][self._string_codes["line_id"][i]],
                "board_mgu": self._code_tables["board_mgu"][self._string_codes["board_mgu"][i]],
                "alight_mgu": self._code_tables["alight_mgu"][self._string_codes["alight_mgu"][i]],
            }
            for c in self._meta_arrays:
                leg[c] = self._meta_arrays[c][i]
            legs.append(leg)
        return legs
