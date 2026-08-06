"""Tier-2 statistical-equivalence harness (parallelising PRD, ADR-0005).

The harness must (a) extract preservation-critical distributions from a world
HDF5 output, (b) compare two such extractions and flag any statistically
significant shift, and (c) round-trip a baseline to disk. Tests exercise the
public interface only, so they survive changes to how the tests are computed.
"""

import h5py
import numpy as np

from may.verification.statistical_equivalence import (
    Distribution,
    compare_distributions,
    extract_distributions,
    format_report,
    load_baseline,
    save_baseline,
)


def _write_world(path):
    """Build a minimal world HDF5 with known distributions.

    Two geo units A(id=10), B(id=20). Venue types [household, company].
    Households 100(A,size2), 101(A,size3), 102(B,size4); company 200(A,size5).
    Nine people with a work_mode property and ages, split across A and B.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("metadata/registries/venue_types",
                         data=np.array([b"household", b"company"]))
        f.create_dataset("geography/ids", data=np.array([10, 20], dtype=np.int64))
        f.create_dataset("metadata/names/geography", data=np.array([b"A", b"B"]))

        # venues: 3 households + 1 company
        f.create_dataset("venues/ids", data=np.array([100, 101, 102, 200], dtype=np.int64))
        f.create_dataset("venues/types", data=np.array([0, 0, 0, 1], dtype=np.uint8))
        f.create_dataset("venues/geo_unit_ids", data=np.array([10, 10, 20, 10], dtype=np.int64))

        # subsets: one per venue; company subset (size 5) must be excluded
        f.create_dataset("venues/subsets/venue_ids",
                         data=np.array([100, 101, 102, 200], dtype=np.int64))
        f.create_dataset("venues/subsets/member_counts",
                         data=np.array([2, 3, 4, 5], dtype=np.int64))

        # population: 9 people, work_mode + ages, geo A(6) / B(3)
        f.create_dataset("population/ids", data=np.arange(9, dtype=np.int64))
        f.create_dataset("population/geo_unit_ids",
                         data=np.array([10, 10, 10, 10, 10, 10, 20, 20, 20], dtype=np.int64))
        f.create_dataset("population/ages",
                         data=np.array([5, 10, 20, 30, 40, 80, 25, 35, 60], dtype=np.int64))
        f.create_dataset("population/properties/work_mode",
                         data=np.array([b"car", b"car", b"car", b"walk", b"walk",
                                        b"bus", b"car", b"walk", b"bus"]))
        # a per-person relational field (unique per person) that must NOT
        # become a proportion metric
        f.create_dataset("population/properties/friendships",
                         data=np.array([f"p{i}".encode() for i in range(9)]))
    return path


def _int_counts(dist):
    return {int(k): v for k, v in dist.counts.items()}


class TestExtractDistributions:

    def test_household_size_excludes_non_household_subsets(self, tmp_path):
        world = _write_world(tmp_path / "world.h5")

        dists = extract_distributions(world)

        assert dists["household_size"].kind == "ordinal"
        assert _int_counts(dists["household_size"]) == {2: 1, 3: 1, 4: 1}

    def test_attribute_proportions_are_nominal(self, tmp_path):
        world = _write_world(tmp_path / "world.h5")

        dists = extract_distributions(world)

        wm = dists["attribute:work_mode"]
        assert wm.kind == "nominal"
        assert wm.counts == {"car": 4, "walk": 3, "bus": 2}

    def test_high_cardinality_relational_field_excluded(self, tmp_path):
        """Per-person relational fields (friendships) have near-unique values;
        they are not proportion metrics and must be dropped by the cardinality
        guard, while genuine low-cardinality attributes are kept."""
        world = _write_world(tmp_path / "world.h5")

        dists = extract_distributions(world, max_attribute_cardinality=3)

        assert "attribute:work_mode" in dists       # 3 distinct -> kept
        assert "attribute:friendships" not in dists  # 9 distinct -> dropped

    def test_age_distribution_split_per_geo_unit(self, tmp_path):
        world = _write_world(tmp_path / "world.h5")

        dists = extract_distributions(world)

        assert dists["age_by_geo:A"].kind == "ordinal"
        assert _int_counts(dists["age_by_geo:A"]) == {5: 1, 10: 1, 20: 1, 30: 1, 40: 1, 80: 1}
        assert _int_counts(dists["age_by_geo:B"]) == {25: 1, 35: 1, 60: 1}


class TestCompareIdentical:

    def test_identical_nominal_distributions_pass(self):
        """Comparing a distribution against itself must not flag a shift —
        this is the baseline-vs-baseline self-comparison the gate relies on."""
        counts = {"car": 5000, "walk": 3000, "public_transport": 2000}
        baseline = {"work_mode": Distribution(kind="nominal", counts=counts)}
        candidate = {"work_mode": Distribution(kind="nominal", counts=dict(counts))}

        report = compare_distributions(baseline, candidate)

        assert report.passed
        assert report.results["work_mode"].verdict == "OK"

    def test_shifted_nominal_distribution_flagged(self):
        """A materially different proportion vector must be flagged and must
        fail the overall report."""
        baseline = {"work_mode": Distribution(
            kind="nominal", counts={"car": 5000, "walk": 3000, "public_transport": 2000})}
        candidate = {"work_mode": Distribution(
            kind="nominal", counts={"car": 2000, "walk": 3000, "public_transport": 5000})}

        report = compare_distributions(baseline, candidate)

        assert not report.passed
        assert report.results["work_mode"].verdict == "SHIFTED"


class TestCompareOrdinal:
    """Ordered numeric metrics (household size, age) use a KS two-sample test
    reconstructed from the value->count histogram."""

    def test_identical_ordinal_distributions_pass(self):
        counts = {1: 4000, 2: 6000, 3: 5000, 4: 3000, 5: 1200, 6: 400}
        baseline = {"household_size": Distribution(kind="ordinal", counts=counts)}
        candidate = {"household_size": Distribution(kind="ordinal", counts=dict(counts))}

        report = compare_distributions(baseline, candidate)

        assert report.passed
        assert report.results["household_size"].test == "ks"
        assert report.results["household_size"].verdict == "OK"

    def test_shifted_ordinal_distribution_flagged(self):
        """A world skewed towards larger households must be flagged."""
        baseline = {"household_size": Distribution(
            kind="ordinal", counts={1: 4000, 2: 6000, 3: 5000, 4: 3000, 5: 1200, 6: 400})}
        candidate = {"household_size": Distribution(
            kind="ordinal", counts={1: 400, 2: 1200, 3: 3000, 4: 5000, 5: 6000, 6: 4000})}

        report = compare_distributions(baseline, candidate)

        assert not report.passed
        assert report.results["household_size"].verdict == "SHIFTED"


class TestStructuralMismatch:

    def test_metric_missing_from_candidate_is_flagged_not_crashed(self):
        """If a parallel bug drops a geo unit's metric, the gate must fail
        loud with a clear verdict rather than raising KeyError."""
        baseline = {
            "work_mode": Distribution(kind="nominal", counts={"car": 10, "walk": 5}),
            "age_by_geo:A": Distribution(kind="ordinal", counts={20: 3, 40: 2}),
        }
        candidate = {
            "work_mode": Distribution(kind="nominal", counts={"car": 10, "walk": 5}),
        }

        report = compare_distributions(baseline, candidate)

        assert not report.passed
        assert report.results["age_by_geo:A"].verdict == "MISSING"


class TestGateWidth:
    """A pass licenses a claim about the metrics that could actually move.

    Metrics whose inputs no in-scope stage writes agree in every run, so a
    total count overstates the evidence: 436 tied metrics and 5 compared ones
    is a 5-metric gate. The report separates the two so the headline count
    cannot be mistaken for the gate's width.
    """

    def test_identical_metrics_are_reported_as_tied(self):
        counts = {"car": 5000, "walk": 3000}
        baseline = {"work_mode": Distribution(kind="nominal", counts=counts)}
        candidate = {"work_mode": Distribution(kind="nominal", counts=dict(counts))}

        report = compare_distributions(baseline, candidate)

        assert report.tied_count == 1
        assert report.compared_count == 0
        assert report.results["work_mode"].tied

    def test_differing_metrics_are_reported_as_compared(self):
        baseline = {
            "work_mode": Distribution(kind="nominal", counts={"car": 5000, "walk": 3000}),
            "age_by_geo:A": Distribution(kind="ordinal", counts={20: 3, 40: 2}),
        }
        candidate = {
            "work_mode": Distribution(kind="nominal", counts={"car": 4000, "walk": 4000}),
            "age_by_geo:A": Distribution(kind="ordinal", counts={20: 3, 40: 2}),
        }

        report = compare_distributions(baseline, candidate)

        assert report.compared_count == 1
        assert report.tied_count == 1
        assert report.results["work_mode"].tied is False
        assert report.results["age_by_geo:A"].tied

    def test_tie_is_reported_but_does_not_change_the_verdict(self):
        """Classification is descriptive only — a tie is still tested and
        still carries a verdict, so `passed` means exactly what it did."""
        counts = {1: 4000, 2: 6000}
        baseline = {"household_size": Distribution(kind="ordinal", counts=counts)}
        candidate = {"household_size": Distribution(kind="ordinal", counts=dict(counts))}

        report = compare_distributions(baseline, candidate)

        assert report.passed
        assert report.results["household_size"].verdict == "OK"
        assert report.results["household_size"].test == "ks"

    def test_missing_metric_is_neither_tied_nor_compared(self):
        """A metric absent from the candidate was never put to a test, so it
        cannot count towards either side of the width."""
        baseline = {
            "work_mode": Distribution(kind="nominal", counts={"car": 10, "walk": 5}),
            "age_by_geo:A": Distribution(kind="ordinal", counts={20: 3}),
        }
        candidate = {
            "work_mode": Distribution(kind="nominal", counts={"car": 10, "walk": 5}),
        }

        report = compare_distributions(baseline, candidate)

        assert report.tied_count == 1
        assert report.compared_count == 0
        assert report.results["age_by_geo:A"].tied is False

    def test_baseline_reloaded_from_disk_still_ties_against_extraction(self, tmp_path):
        """`save_baseline` stringifies histogram keys, so a baseline read back
        from JSON meets an in-memory extraction with differently-typed keys.
        A tie must be recognised through that, or every ordinal metric in a
        real gate run would be misreported as compared."""
        distributions = extract_distributions(_write_world(tmp_path / "world.h5"))
        path = tmp_path / "baseline.json"
        save_baseline(distributions, path)

        report = compare_distributions(load_baseline(path), distributions)

        assert report.compared_count == 0
        assert report.tied_count == len(distributions)

    def test_report_states_the_width_not_just_the_count(self, tmp_path):
        baseline = {
            "work_mode": Distribution(kind="nominal", counts={"car": 5000, "walk": 3000}),
            "age_by_geo:A": Distribution(kind="ordinal", counts={20: 3, 40: 2}),
        }
        candidate = {
            "work_mode": Distribution(kind="nominal", counts={"car": 4000, "walk": 4000}),
            "age_by_geo:A": Distribution(kind="ordinal", counts={20: 3, 40: 2}),
        }

        summary = format_report(compare_distributions(baseline, candidate))

        assert "1 compared" in summary
        assert "1 tied" in summary


class TestBaselineRoundTrip:

    def test_saved_baseline_loads_and_self_compares_cleanly(self, tmp_path):
        """A captured baseline reloaded from disk must compare cleanly against
        itself — the reproducible-artefact gate depends on this."""
        distributions = {
            "work_mode": Distribution(
                kind="nominal", counts={"car": 5000, "walk": 3000}),
            "household_size": Distribution(
                kind="ordinal", counts={1: 4000, 2: 6000, 3: 5000}),
        }
        path = tmp_path / "baseline.json"

        save_baseline(distributions, path)
        loaded = load_baseline(path)

        report = compare_distributions(loaded, loaded)
        assert report.passed
        assert set(loaded) == {"work_mode", "household_size"}
        assert loaded["household_size"].kind == "ordinal"
