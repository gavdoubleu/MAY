"""`Slice` types: stable, seedable sub-partitions of a `Cohort`.

A Slice answers "which worker processes this, and what stable identity seeds
its RNG stream?" (CONTEXT.md). Its `digest` is derived from the stable domain
identity of its members — never from position or size — so the same Cohort can
be re-sliced differently across runs without changing results (ADR-0002).
"""
import hashlib
from dataclasses import dataclass
from typing import List, Sequence


def _digest_identities(identities: Sequence[str]) -> int:
    """Platform-independent, order-independent 64-bit digest of stable member
    identities. Members are sorted first so a Slice's seed depends on *which*
    members it holds, not the order they were listed in.
    """
    hasher = hashlib.blake2b(digest_size=8)
    for identity in sorted(identities):
        hasher.update(identity.encode('utf-8'))
        hasher.update(b'\x00')  # unambiguous delimiter between members
    return int.from_bytes(hasher.digest(), 'big')


class Slice:
    """Base Slice: exposes a stable `digest` used to seed its RNG stream."""

    @property
    def digest(self) -> int:
        raise NotImplementedError


@dataclass
class GeoUnitSlice(Slice):
    """A geo_unit-axis Slice: a list of geo_unit codes (ONS names, e.g.
    'E00004320'). The code is stable across runs and load order, unlike the
    runtime-generated numeric unit id (ADR-0002).
    """
    unit_codes: List[str]

    @property
    def digest(self) -> int:
        return _digest_identities(self.unit_codes)


@dataclass
class PopulationSlice(Slice):
    """A person-axis Slice: a set of stable person-ids. Order-independent by
    identity — the same id set seeds the same stream however it is listed
    (ADR-0002).
    """
    person_ids: Sequence[int]

    @property
    def digest(self) -> int:
        return _digest_identities([str(person_id) for person_id in self.person_ids])
