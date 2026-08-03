"""The composed-pipeline bit-for-bit gate for Phase 4b.

Captured on the pre-fix HEAD. Any change to it means the world changed, which
for this phase is a bug by definition — see `golden_fingerprint` for what is
hashed and what it does not cover.
"""
import pytest

from tests.test_integration.golden_fingerprint import (
    LOOP_LEVELS, load_golden, run_and_fingerprint)


@pytest.mark.parametrize("loop_level", LOOP_LEVELS)
def test_allocated_world_is_unchanged(loop_level, tmp_path):
    golden = load_golden()
    assert run_and_fingerprint(loop_level, str(tmp_path)) == golden[loop_level]
