"""Tier-2 statistical-equivalence harness for the parallelising work (ADR-0005).

Per-Slice RNG seeding changes the exact world output (accepted, not a
regression); what must be preserved is the *distributions*. This module

  1. extracts preservation-critical distributions from a world HDF5 output,
  2. compares two extractions and flags any statistically significant shift,
  3. round-trips a baseline to disk (compact value->count histograms),

so a pre-parallel baseline can gate every later parallelisation phase.

A metric is a `Distribution`: a value->count histogram plus a `kind`:
  - ``nominal``  – unordered categories (attribute proportions) -> chi-square.
  - ``ordinal``  – ordered numeric bins (household size, age)    -> KS.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from scipy import stats


@dataclass
class Distribution:
    """A single metric as a value->count histogram.

    Stored as counts (not raw samples) so a baseline stays compact and
    git-diffable while still reconstructing the empirical distribution exactly
    for the discrete quantities this harness compares.
    """

    kind: str            # "nominal" | "ordinal"
    counts: Dict         # value -> count


@dataclass
class MetricResult:
    """Outcome of comparing one metric between baseline and candidate."""

    test: str
    statistic: float
    pvalue: float
    verdict: str         # "OK" | "SHIFTED" | "MISSING"
    tied: bool = False   # baseline and candidate histograms were identical


@dataclass
class ComparisonReport:
    """Aggregate verdict across all metrics.

    Carries the gate's *width* alongside its metric count. A metric whose
    inputs no in-scope stage writes is identical in every run and contributes
    no evidence, so a bare count overstates what a pass licenses — hence
    `compared_count`, the metrics that actually had something to disagree
    about, reported next to it.
    """

    passed: bool
    results: Dict[str, MetricResult] = field(default_factory=dict)

    @property
    def tied_count(self) -> int:
        """Metrics whose baseline and candidate histograms were identical."""
        return sum(1 for result in self.results.values() if result.tied)

    @property
    def compared_count(self) -> int:
        """Metrics that differed, so the test performed carries information.

        Excludes MISSING metrics: absent from the candidate, they were never
        put to a test and belong to neither side of the width.
        """
        return sum(1 for result in self.results.values()
                   if not result.tied and result.verdict != "MISSING")


def _compare_nominal(baseline: Distribution, candidate: Distribution,
                     alpha: float) -> MetricResult:
    """Chi-square test of independence on the aligned 2xk count table."""
    labels = sorted(set(baseline.counts) | set(candidate.counts), key=str)
    base_row = [baseline.counts.get(label, 0) for label in labels]
    cand_row = [candidate.counts.get(label, 0) for label in labels]
    statistic, pvalue, _, _ = stats.chi2_contingency([base_row, cand_row])
    verdict = "SHIFTED" if pvalue < alpha else "OK"
    return MetricResult("chi2", float(statistic), float(pvalue), verdict)


def _samples_from_counts(counts: Dict) -> np.ndarray:
    """Reconstruct the sample array of an ordinal histogram (exact ECDF)."""
    values = np.array([float(v) for v in counts.keys()], dtype=np.float64)
    freqs = np.array([int(c) for c in counts.values()], dtype=np.int64)
    return np.repeat(values, freqs)


def _compare_ordinal(baseline: Distribution, candidate: Distribution,
                     alpha: float) -> MetricResult:
    """Two-sample Kolmogorov-Smirnov test on the reconstructed samples."""
    base_samples = _samples_from_counts(baseline.counts)
    cand_samples = _samples_from_counts(candidate.counts)
    result = stats.ks_2samp(base_samples, cand_samples)
    verdict = "SHIFTED" if result.pvalue < alpha else "OK"
    return MetricResult("ks", float(result.statistic), float(result.pvalue), verdict)


_COMPARATORS = {"nominal": _compare_nominal, "ordinal": _compare_ordinal}


def _normalised_counts(distribution: Distribution) -> Dict:
    """Histogram with keys cast the way this metric's comparator reads them.

    `save_baseline` stringifies every key, so a baseline loaded from disk meets
    an in-memory extraction with differently-typed keys for the same value.
    Normalising the way the comparators already do — labels for nominal, floats
    for ordinal — is what lets a genuine tie be recognised across that boundary.
    """
    caster = str if distribution.kind == "nominal" else float
    return {caster(value): int(count) for value, count in distribution.counts.items()}


def _is_tie(baseline: Distribution, candidate: Distribution) -> bool:
    """True when the two histograms are identical, so nothing could disagree."""
    return _normalised_counts(baseline) == _normalised_counts(candidate)


def compare_distributions(baseline: Dict[str, Distribution],
                          candidate: Dict[str, Distribution],
                          alpha: float = 0.05) -> ComparisonReport:
    """Compare two metric maps; flag metrics with a significant shift.

    Args:
        baseline / candidate: metric-name -> `Distribution`.
        alpha: per-metric significance threshold.

    Returns:
        `ComparisonReport` whose ``passed`` is True iff no metric shifted.
    """
    results: Dict[str, MetricResult] = {}
    for name, base_dist in baseline.items():
        if name not in candidate:
            results[name] = MetricResult("none", float("nan"), 0.0, "MISSING")
            continue
        comparator = _COMPARATORS[base_dist.kind]
        result = comparator(base_dist, candidate[name], alpha)
        # Classified after testing, not instead of it: a tie is still put to
        # the test so every metric's verdict is reached the same way.
        result.tied = _is_tie(base_dist, candidate[name])
        results[name] = result

    passed = all(r.verdict == "OK" for r in results.values())
    return ComparisonReport(passed=passed, results=results)


def _decode(array) -> np.ndarray:
    """Return a string view of a byte/str/numeric HDF5 array."""
    return array.astype(str)


def _household_size_distribution(f) -> Dict[str, Distribution]:
    """Member-count histogram of household subsets only (excludes other venues)."""
    if "venues" not in f or "subsets" not in f["venues"]:
        return {}
    registry = list(_decode(f["metadata/registries/venue_types"][:]))
    if "household" not in registry:
        return {}
    household_type_id = registry.index("household")

    venue_ids = f["venues/ids"][:]
    venue_types = f["venues/types"][:]
    household_venue_ids = venue_ids[venue_types == household_type_id]

    subset_venue_ids = f["venues/subsets/venue_ids"][:]
    member_counts = f["venues/subsets/member_counts"][:]
    is_household = np.isin(subset_venue_ids, household_venue_ids)
    sizes = member_counts[is_household]

    counts = {int(size): int(freq)
              for size, freq in Counter(sizes.tolist()).items()}
    return {"household_size": Distribution(kind="ordinal", counts=counts)}


def _attribute_proportion_distributions(f, max_cardinality) -> Dict[str, Distribution]:
    """One nominal proportion metric per person property.

    Per-person relational/structural fields (e.g. friendship or partner IDs)
    have near-unique values and are not proportion metrics; fields whose
    distinct-value count exceeds ``max_cardinality`` are skipped.
    """
    if "population" not in f or "properties" not in f["population"]:
        return {}
    dists: Dict[str, Distribution] = {}
    for prop_name in f["population/properties"].keys():
        values = _decode(f["population/properties"][prop_name][:])
        counts = {str(label): int(freq)
                  for label, freq in Counter(values.tolist()).items()}
        if len(counts) > max_cardinality:
            continue
        dists[f"attribute:{prop_name}"] = Distribution(kind="nominal", counts=counts)
    return dists


def _age_by_geo_unit_distributions(f) -> Dict[str, Distribution]:
    """One ordinal age histogram per geo unit the population sits in.

    Companies carry no age property, so the geo-scoped age quantity the
    in-scope stages perturb is the population's own age composition per unit.
    """
    if "population" not in f or "ages" not in f["population"]:
        return {}
    ages = f["population/ages"][:]
    geo_ids = f["population/geo_unit_ids"][:]
    id_to_name = dict(zip(f["geography/ids"][:].tolist(),
                          _decode(f["metadata/names/geography"][:]).tolist()))

    dists: Dict[str, Distribution] = {}
    for geo_id in np.unique(geo_ids):
        unit_name = id_to_name.get(int(geo_id))
        if unit_name is None:
            continue
        unit_ages = ages[geo_ids == geo_id]
        counts = {int(age): int(freq)
                  for age, freq in Counter(unit_ages.tolist()).items()}
        dists[f"age_by_geo:{unit_name}"] = Distribution(kind="ordinal", counts=counts)
    return dists


def extract_distributions(hdf5_path,
                          max_attribute_cardinality: int = 200) -> Dict[str, Distribution]:
    """Extract the preservation-critical metric map from a world HDF5 output.

    Covers the three families the in-scope parallelised stages perturb:
    household-size (allocation), per-attribute proportions (attribute
    assignment), and population age per geo unit.

    Args:
        hdf5_path: path to an exported world HDF5.
        max_attribute_cardinality: person properties with more distinct values
            than this are skipped (relational/structural fields, not
            proportion metrics).
    """
    distributions: Dict[str, Distribution] = {}
    with h5py.File(hdf5_path, "r") as f:
        distributions.update(_household_size_distribution(f))
        distributions.update(
            _attribute_proportion_distributions(f, max_attribute_cardinality))
        distributions.update(_age_by_geo_unit_distributions(f))
    return distributions


def save_baseline(distributions: Dict[str, Distribution], path) -> None:
    """Persist a metric map as a compact, git-diffable JSON histogram file."""
    payload = {
        name: {"kind": dist.kind,
               "counts": {str(value): int(count)
                          for value, count in dist.counts.items()}}
        for name, dist in distributions.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_baseline(path) -> Dict[str, Distribution]:
    """Load a metric map previously written by `save_baseline`.

    Histogram keys are read back as strings; the nominal path treats them as
    labels and the ordinal path casts them to float, so either kind round-trips.
    """
    payload = json.loads(Path(path).read_text())
    return {
        name: Distribution(kind=entry["kind"], counts=dict(entry["counts"]))
        for name, entry in payload.items()
    }


def format_report(report: ComparisonReport, only_shifted: bool = True) -> str:
    """Human-readable summary; by default lists only the metrics that shifted."""
    missing_count = sum(1 for result in report.results.values()
                        if result.verdict == "MISSING")
    width = (f"{len(report.results)} metrics: {report.compared_count} compared, "
             f"{report.tied_count} tied (identical histograms)")
    if missing_count:
        width += f", {missing_count} missing"
    lines = [f"Tier-2 statistical equivalence: "
             f"{'PASS' if report.passed else 'FAIL'} ({width})"]
    for name, result in sorted(report.results.items()):
        if only_shifted and result.verdict == "OK":
            continue
        lines.append(f"  [{result.verdict}] {name}: {result.test} "
                     f"stat={result.statistic:.4g} p={result.pvalue:.3g}")
    return "\n".join(lines)


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Tier-2 statistical-equivalence harness (ADR-0005).")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="extract a baseline from a world HDF5")
    cap.add_argument("world_hdf5")
    cap.add_argument("baseline_json")

    cmp_ = sub.add_parser("compare",
                          help="compare a candidate world against a baseline")
    cmp_.add_argument("baseline_json")
    cmp_.add_argument("candidate_hdf5")
    cmp_.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args(argv)

    if args.command == "capture":
        distributions = extract_distributions(args.world_hdf5)
        save_baseline(distributions, args.baseline_json)
        print(f"Captured {len(distributions)} metrics -> {args.baseline_json}")
        return 0

    baseline = load_baseline(args.baseline_json)
    candidate = extract_distributions(args.candidate_hdf5)
    report = compare_distributions(baseline, candidate, alpha=args.alpha)
    print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(_main())
