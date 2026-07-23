"""Process-based Slice parallelism substrate (parallelise-easy-stuff, Phase 2).

Public interface: `parallel_map` plus the `Slice` types. Everything else is
internal and may change without notice.
"""
from .pool import parallel_map, resolve_worker_count
from .slicing import GeoUnitSlice, PopulationSlice, Slice

__all__ = ['parallel_map', 'resolve_worker_count',
           'Slice', 'GeoUnitSlice', 'PopulationSlice']
