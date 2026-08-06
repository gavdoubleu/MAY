"""
Unit tests for data loading and lookup logic.

Covers:
- _normalize_probabilities(): all-zeros, already-normalized, negative values
- GeoDistributionSource: CSV parsing, geo_unit filtering, fallback
- PairProbabilitySource: nested lookup, uniform fallback
- MultiKeyLookupSource: multi-key resolution, category_lookup, ancestor_lookup, caching
- OriginDestinationMatrixSource: destination exclusion, likelihood normalization
- GUSamplerSource: weight-based normalization, exclude_rows filtering
- DataSourceManager._initialize_sources(): routing logic
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
from io import StringIO

from may.attribute_assignment.data_sources import (
    DataSource,
    GeoDistributionSource,
    DiversitySource,
    PairProbabilitySource,
    MultiKeyLookupSource,
    OriginDestinationMatrixSource,
    GUSamplerSource,
    DataSourceManager,
    _ordered_key_columns,
    _require_declared_columns,
    MissingDeclaredColumns,
)


# _ordered_key_columns: canonical key_columns mapping

class TestOrderedKeyColumns:
    def test_single_key_mapping(self):
        assert _ordered_key_columns({"key_columns": {"geo_unit": None}}, "s", expected=1) == ["geo_unit"]

    def test_two_key_mapping_preserves_order(self):
        cfg = {"key_columns": {"geo_unit": None, "first_ethnicity": None}}
        assert _ordered_key_columns(cfg, "s", expected=2) == ["geo_unit", "first_ethnicity"]

    def test_retired_singular_key_column_raises(self):
        with pytest.raises(ValueError, match="'key_column' is retired"):
            _ordered_key_columns({"key_column": "geo_unit"}, "s")

    def test_list_form_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _ordered_key_columns({"key_columns": ["geo_unit", "first_ethnicity"]}, "s")

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="needs 'key_columns'"):
            _ordered_key_columns({}, "s")

    def test_wrong_count_raises(self):
        with pytest.raises(ValueError, match="expected 2 key column"):
            _ordered_key_columns({"key_columns": {"geo_unit": None}}, "s", expected=2)


# Minimal real objects

class MinimalGeoUnit:
    def __init__(self, name, level="SGU", parent=None):
        self.name = name
        self.level = level
        self.parent = parent

    def get_ancestor_by_level(self, level):
        if self.level == level:
            return self
        current = self.parent
        while current is not None:
            if current.level == level:
                return current
            current = current.parent
        return None


class MinimalPerson:
    _next_id = 4000

    def __init__(self, age=30, sex="M", geographical_unit=None, properties=None):
        self.id = MinimalPerson._next_id
        MinimalPerson._next_id += 1
        self.age = age
        self.sex = sex
        self.geographical_unit = geographical_unit
        self.properties = properties if properties is not None else {}


class MinimalHousehold:
    def __init__(self, geographical_unit=None):
        self.geographical_unit = geographical_unit


class MinimalDataSourceConfig:
    """Mimics the config object returned by _parse_data_sources."""
    def __init__(self, source_type, config):
        self.type = source_type
        self.config = config


class MinimalAssignmentConfig:
    """Mimics AttributeAssignmentConfig for MultiKeyLookupSource."""
    def __init__(self, required_attributes=None, categories=None):
        self.required_attributes = required_attributes or {}
        self.categories = categories or {}
        self._category_lookup_cache = {}

    def get_category_for_value(self, value, attr_name):
        """Simple category lookup for testing."""
        for cat_name, cat_config in self.categories.items():
            ranges = cat_config.get('ranges', [])
            for r in ranges:
                if r.get('min', float('-inf')) <= value < r.get('max', float('inf')):
                    return cat_config
        return None


@pytest.fixture(autouse=True)
def reset_ids():
    MinimalPerson._next_id = 4000


# _normalize_probabilities() Tests

class TestNormalizeProbabilities:
    """
    Tests the base DataSource._normalize_probabilities method.
    """

    def _normalize(self, probs):
        """Call the method on a dummy instance."""
        source = GeoDistributionSource("test", {"files": [], "fallback": {}})
        return source._normalize_probabilities(probs)

    def test_already_normalized(self):
        probs = {"W": 0.5, "A": 0.3, "B": 0.2}
        result = self._normalize(probs)
        assert result == probs  # should return unchanged (same object ref)

    def test_not_normalized_scales_correctly(self):
        probs = {"W": 2.0, "A": 3.0, "B": 5.0}
        result = self._normalize(probs)
        assert abs(result["W"] - 0.2) < 1e-10
        assert abs(result["A"] - 0.3) < 1e-10
        assert abs(result["B"] - 0.5) < 1e-10
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_all_zeros_raises(self):
        """All-zero distribution can't be sampled, and there are no fallbacks."""
        probs = {"W": 0.0, "A": 0.0, "B": 0.0}
        with pytest.raises(ValueError, match="All-zero"):
            self._normalize(probs)

    def test_single_entry(self):
        probs = {"W": 5.0}
        result = self._normalize(probs)
        assert abs(result["W"] - 1.0) < 1e-10

    def test_very_small_values(self):
        probs = {"W": 1e-15, "A": 1e-15}
        result = self._normalize(probs)
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_negative_values_clamped_to_zero(self):
        """
        Negative probabilities are invalid for np.random.choice.
        They must be clamped to 0 and a warning logged.
        """
        probs = {"W": -1.0, "A": 3.0, "B": 2.0}
        result = self._normalize(probs)
        # W clamped to 0, then normalized over A(3)+B(2)=5
        assert result["W"] == 0.0
        assert abs(result["A"] - 0.6) < 1e-10
        assert abs(result["B"] - 0.4) < 1e-10
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_negative_values_logs_warning(self, caplog):
        import logging
        probs = {"W": -1.0, "A": 3.0}
        with caplog.at_level(logging.WARNING):
            self._normalize(probs)
        assert any("Negative probability" in msg for msg in caplog.messages)

    def test_preserves_keys(self):
        probs = {"W": 2.0, "A": 3.0}
        result = self._normalize(probs)
        assert set(result.keys()) == {"W", "A"}

    def test_empty_dict_raises(self):
        """Empty distribution → raise. No fallbacks."""
        with pytest.raises(ValueError, match="Empty probability"):
            self._normalize({})

    def test_negative_sum_to_zero_raises(self):
        """
        Negatives clamp to 0; if nothing positive remains the distribution is
        unsampleable, so raise rather than invent a uniform one.
        e.g. {A: -1, B: 0} → clamp → {A: 0, B: 0} → raise.
        """
        probs = {"A": -1.0, "B": 0.0}
        with pytest.raises(ValueError, match="All-zero"):
            self._normalize(probs)

    def test_all_negative_values_raise(self, caplog):
        """All values negative → all clamped to 0 → all-zero → raise."""
        import logging
        probs = {"A": -2.0, "B": -3.0}
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="All-zero"):
                self._normalize(probs)
        # Negatives are still warned about before the raise.
        assert any("Negative probability" in msg for msg in caplog.messages)

    def test_mixed_negative_positive_summing_to_zero(self):
        """
        {A: -1, B: 1} → clamp → {A: 0, B: 1} → normalize → {A: 0, B: 1}
        (Not uniform, because B had a legitimate positive value.)
        """
        probs = {"A": -1.0, "B": 1.0}
        result = self._normalize(probs)
        assert result["A"] == 0.0
        assert abs(result["B"] - 1.0) < 1e-10


# GeoDistributionSource Tests

class TestGeoDistributionSource:
    """Tests CSV-based geographic distribution lookups."""

    def _make_source_with_data(self, lookup_data, fallback=None):
        """Create a source with pre-loaded data (skip CSV reading)."""
        source = GeoDistributionSource("test_geo", {
            "files": [],
            "fallback": fallback or {},
        })
        source._lookup = lookup_data
        source._data_loaded = True
        return source

    def test_lookup_resolves_person_geo_unit(self):
        source = self._make_source_with_data(
            {"E00001": {"W": 0.8, "A": 0.2}}
        )
        person = MinimalPerson(geographical_unit=MinimalGeoUnit("E00001"))
        result = source.lookup(person, None, None)
        assert result == {"W": 0.8, "A": 0.2}

    def test_lookup_prefers_household_geo_over_person(self):
        """Residence venue geo wins over the person's own."""
        source = self._make_source_with_data(
            {"E00001": {"W": 1.0}, "E00002": {"A": 1.0}}
        )
        person = MinimalPerson(geographical_unit=MinimalGeoUnit("E00002"))
        household = MinimalHousehold(geographical_unit=MinimalGeoUnit("E00001"))
        assert source.lookup(person, household, None) == {"W": 1.0}

    def test_lookup_no_geo_unit_raises(self):
        """No residence geo anywhere → hard error, no silent miss."""
        source = self._make_source_with_data({"E00001": {"W": 1.0}})
        person = MinimalPerson(geographical_unit=None)
        with pytest.raises(KeyError, match="no residence geographical_unit"):
            source.lookup(person, None, None)

    def test_lookup_missing_geo_unit_raises(self):
        """No fallbacks: a missing geo unit is a hard error."""
        source = self._make_source_with_data(
            {"E00001": {"W": 0.8, "A": 0.2}},
        )
        person = MinimalPerson(geographical_unit=MinimalGeoUnit("MISSING"))
        with pytest.raises(KeyError, match="no row for geo unit"):
            source.lookup(person, None, None)

    def test_lookup_before_data_loaded_raises(self):
        source = GeoDistributionSource("test_geo", {"files": []})
        # _data_loaded is False by default
        with pytest.raises(RuntimeError, match="Data not loaded"):
            source.lookup("E00001")

    def test_missing_file_fails_loud_at_load(self):
        """A missing data file aborts at load, not a silent empty source that
        explodes later at lookup."""
        source = GeoDistributionSource("test_geo", {
            "files": [{"path": "/no/such/file.csv", "key_columns": {"geo_unit": None}}]
        })
        with pytest.raises(FileNotFoundError, match="not found"):
            source.load_data()

    @staticmethod
    def _file_entry(path, rows):
        path.write_text("geo_unit,W,A\n" + "\n".join(rows) + "\n")
        return {
            "path": str(path),
            "key_columns": {"geo_unit": None},
            "value_columns": {"W": "W", "A": "A"},
        }

    def test_multiple_files_merge_disjoint_keys(self, tmp_path):
        """Per-nation files with disjoint geo units combine into one lookup."""
        source = GeoDistributionSource("test_geo", {"files": [
            self._file_entry(tmp_path / "a.csv", ["E00001,8,2"]),
            self._file_entry(tmp_path / "b.csv", ["N00001,1,9"]),
        ]})
        source.load_data()
        assert set(source._lookup) == {"E00001", "N00001"}
        assert abs(source._lookup["N00001"]["A"] - 0.9) < 1e-9

    def test_overlapping_keys_across_files_raise(self, tmp_path):
        """The same geo unit in two files must fail loud, never last-file-wins."""
        source = GeoDistributionSource("test_geo", {"files": [
            self._file_entry(tmp_path / "a.csv", ["E00001,8,2"]),
            self._file_entry(tmp_path / "b.csv", ["E00001,1,9"]),
        ]})
        with pytest.raises(ValueError, match="already loaded"):
            source.load_data()

    def test_parse_dataframe_with_total_column(self):
        """When total_column is provided, values are divided by total first."""
        source = GeoDistributionSource("test", {
            "files": [],
            "fallback": {},
        })
        df = pd.DataFrame({
            "geo_unit": ["E00001", "E00002"],
            "white": [80, 50],
            "asian": [20, 50],
            "total": [100, 100],
        })
        result = source._parse_dataframe(
            df, "geo_unit",
            {"W": "white", "A": "asian"},
            total_column="total"
        )
        assert abs(result["E00001"]["W"] - 0.8) < 1e-10
        assert abs(result["E00001"]["A"] - 0.2) < 1e-10
        assert abs(result["E00002"]["W"] - 0.5) < 1e-10

    def test_parse_dataframe_without_total_column(self):
        """Without total_column, raw values are normalized."""
        source = GeoDistributionSource("test", {
            "files": [],
            "fallback": {},
        })
        df = pd.DataFrame({
            "geo_unit": ["E00001"],
            "white": [80.0],
            "asian": [20.0],
        })
        result = source._parse_dataframe(
            df, "geo_unit",
            {"W": "white", "A": "asian"},
        )
        assert abs(result["E00001"]["W"] - 0.8) < 1e-10
        assert abs(result["E00001"]["A"] - 0.2) < 1e-10


# PairProbabilitySource Tests

class TestPairProbabilitySource:
    """Tests conditional pair probability lookups."""

    def _make_source_with_data(self, lookup_data, fallback_type='uniform'):
        source = PairProbabilitySource("test_pair", {
            "files": [],
            "fallback": fallback_type,
        })
        source._lookups = lookup_data
        source._data_loaded = True
        return source

    def test_lookup_existing_geo_and_value(self):
        source = self._make_source_with_data({
            "E00001": {
                "W": {"W": 0.9, "A": 0.1},
                "A": {"W": 0.1, "A": 0.9},
            }
        })
        result = source.lookup("E00001", "W")
        assert result["W"] == 0.9

    def test_lookup_missing_first_value_raises(self):
        """No fallbacks: a missing (geo, first-value) pair is an error."""
        source = self._make_source_with_data({
            "E00001": {"W": {"W": 0.9, "A": 0.1}}
        })
        with pytest.raises(KeyError, match="no pair row"):
            source.lookup("E00001", "B")  # "B" not in data

    def test_lookup_missing_geo_unit_raises(self):
        source = self._make_source_with_data({
            "E00001": {"W": {"W": 0.9, "A": 0.1}}
        })
        with pytest.raises(KeyError, match="no pair row"):
            source.lookup("MISSING", "W")

    def test_lookup_before_data_loaded_raises(self):
        source = PairProbabilitySource("test_pair", {"files": []})
        with pytest.raises(RuntimeError, match="Data not loaded"):
            source.lookup("E00001", "W")


# OriginDestinationMatrixSource Tests

def write_od_csv(path, origin, destination, likelihood, extra_columns=None):
    """extra_columns: Dict[column_name, List[value]] aligned with the rows."""
    data = {"origin": origin, "destination": destination, "likelihood": likelihood}
    if extra_columns:
        data.update(extra_columns)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def make_od_config(csv_path, out_of_boundary="error", metadata_columns=None,
                    exclude_destinations=None, on_empty=None, outside_value=None,
                    redistributed_flag=None, chunk_size=None, destination_level=None,
                    sentinel_preserves=None):
    """csv_path: one path, or a list of paths for a multi-file source.
    metadata_columns: one mapping applied to every file, or a list of mappings
    aligned with csv_path."""
    csv_paths = csv_path if isinstance(csv_path, (list, tuple)) else [csv_path]
    if isinstance(metadata_columns, (list, tuple)):
        per_file_metadata = list(metadata_columns)
    else:
        per_file_metadata = [metadata_columns or {}] * len(csv_paths)
    file_configs = []
    for path, file_metadata in zip(csv_paths, per_file_metadata):
        file_config = {
            "path": str(path),
            "key_columns": {"origin": None},
            "destination_column": "destination",
            "likelihood_column": "likelihood",
            "metadata_columns": file_metadata or {},
            "exclude_destinations": exclude_destinations or [],
        }
        if destination_level is not None:
            file_config["destination_level"] = destination_level
        file_configs.append(file_config)
    config = {"files": file_configs, "out_of_boundary": out_of_boundary}
    if on_empty is not None:
        config["on_empty"] = on_empty
    if outside_value is not None:
        config["outside_value"] = outside_value
    if sentinel_preserves is not None:
        config["sentinel_preserves"] = sentinel_preserves
    if redistributed_flag is not None:
        config["redistributed_flag"] = redistributed_flag
    if chunk_size is not None:
        config["chunk_size"] = chunk_size
    return config


class TestOriginDestinationMatrixSource:
    """Tests O-D matrix loading and lookup, via the public load_data()/lookup()
    interface only — never the internal columnar/array layout."""

    def test_lookup_existing_origin(self, tmp_path):
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["ORIGIN_A", "ORIGIN_A"], ["DEST_1", "DEST_2"], [0.7, 0.3],
            extra_columns={"mode": ["car", "bus"]},
        )
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, metadata_columns={"mode": "mode"})
        )
        source.load_data()
        result = source.lookup("ORIGIN_A")
        assert len(result) == 2
        assert result[0][0] == "DEST_1"
        assert abs(result[0][2] - 0.7) < 1e-10
        assert result[0][1]["mode"] == "car"

    def test_lookup_missing_origin_raises(self, tmp_path):
        csv_path = write_od_csv(tmp_path / "od.csv", ["ORIGIN_A"], ["DEST_1"], [1.0])
        source = OriginDestinationMatrixSource("test_od", make_od_config(csv_path))
        source.load_data()
        with pytest.raises(KeyError, match="no destinations for origin"):
            source.lookup("MISSING")

    def test_lookup_before_data_loaded_raises(self):
        source = OriginDestinationMatrixSource("test_od", {"files": [], "out_of_boundary": "error"})
        with pytest.raises(RuntimeError, match="Data not loaded"):
            source.lookup("ORIGIN_A")

    def test_destination_exclusion(self, tmp_path):
        """Excluded destinations should not appear in lookup results."""
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A", "A", "A"], ["D1", "D2", "D3"], [5.0, 3.0, 2.0]
        )
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, exclude_destinations=["D2"])
        )
        source.load_data()
        destinations = [d[0] for d in source.lookup("A")]
        assert "D1" in destinations
        assert "D2" not in destinations
        assert "D3" in destinations

    def test_likelihood_normalization(self, tmp_path):
        """Likelihoods should be normalized to sum to 1.0 per origin."""
        csv_path = write_od_csv(tmp_path / "od.csv", ["A", "A"], ["D1", "D2"], [3.0, 7.0])
        source = OriginDestinationMatrixSource("test_od", make_od_config(csv_path))
        source.load_data()
        result = source.lookup("A")
        total = sum(lik for _, _, lik in result)
        assert abs(total - 1.0) < 1e-10
        by_dest = {d: lik for d, _, lik in result}
        assert abs(by_dest["D1"] - 0.3) < 1e-10
        assert abs(by_dest["D2"] - 0.7) < 1e-10

    def test_metadata_columns_collected(self, tmp_path):
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A"], ["D1"], [1.0],
            extra_columns={"mode": ["car"], "distance": [10.5]},
        )
        source = OriginDestinationMatrixSource(
            "test_od",
            make_od_config(csv_path, metadata_columns={"mode": "mode", "distance": "distance"}),
        )
        source.load_data()
        meta = source.lookup("A")[0][1]
        assert meta["mode"] == "car"
        assert meta["distance"] == 10.5

    def test_multiple_origins(self, tmp_path):
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A", "A", "B"], ["D1", "D2", "D3"], [0.6, 0.4, 1.0]
        )
        source = OriginDestinationMatrixSource("test_od", make_od_config(csv_path))
        source.load_data()
        assert len(source.lookup("A")) == 2
        assert len(source.lookup("B")) == 1

    def test_exclusion_with_all_destinations_excluded(self, tmp_path):
        """If all destinations are excluded, origin has empty list, not a
        missing key."""
        csv_path = write_od_csv(tmp_path / "od.csv", ["A"], ["D1"], [1.0])
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, exclude_destinations=["D1"])
        )
        source.load_data()
        assert source.lookup("A") == []

    def test_missing_file_fails_loud_at_load(self):
        source = OriginDestinationMatrixSource("test_od", {
            "files": [{"path": "/no/such/file.csv", "key_columns": {"origin": None},
                       "destination_column": "destination", "likelihood_column": "likelihood"}],
            "out_of_boundary": "error",
        })
        with pytest.raises(FileNotFoundError, match="not found"):
            source.load_data()

    def test_chunking_splits_single_origin_across_chunks(self, tmp_path):
        """An origin's rows split across chunk boundaries must still end up
        grouped together under one lookup() call."""
        origins = ["A"] * 5
        destinations = [f"D{i}" for i in range(5)]
        likelihoods = [1.0] * 5
        csv_path = write_od_csv(tmp_path / "od.csv", origins, destinations, likelihoods)
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, chunk_size=2)
        )
        source.load_data()
        result = source.lookup("A")
        assert {d for d, _, _ in result} == set(destinations)

    def test_geo_units_filter_restricts_origins(self, tmp_path):
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A", "B"], ["D1", "D2"], [1.0, 1.0]
        )
        source = OriginDestinationMatrixSource("test_od", make_od_config(csv_path))
        source.load_data(geo_units={"A", "D1", "D2"})
        assert source.lookup("A") is not None
        with pytest.raises(KeyError):
            source.lookup("B")


class TestOriginDestinationMultiFile:
    """One source, several files covering disjoint origins (the production
    shape: England+Wales and Scotland in separate published tables). Every
    file's origins must survive into one lookup."""

    def test_origins_from_all_files_resolve(self, tmp_path):
        first = write_od_csv(tmp_path / "first.csv", ["A", "A"], ["D1", "D2"], [3.0, 7.0],
                             extra_columns={"mode": ["car", "bus"]})
        second = write_od_csv(tmp_path / "second.csv", ["B"], ["D3"], [1.0],
                              extra_columns={"mode": ["car"]})
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config([first, second], metadata_columns={"mode": "mode"})
        )
        source.load_data()
        assert {d for d, _, _ in source.lookup("A")} == {"D1", "D2"}
        assert source.lookup("B") == [("D3", {"mode": "car"}, 1.0)]

    def test_normalisation_is_per_origin_not_per_file(self, tmp_path):
        first = write_od_csv(tmp_path / "first.csv", ["A", "A"], ["D1", "D2"], [3.0, 7.0])
        second = write_od_csv(tmp_path / "second.csv", ["B", "B"], ["D1", "D2"], [1.0, 1.0])
        source = OriginDestinationMatrixSource("test_od", make_od_config([first, second]))
        source.load_data()
        for origin in ("A", "B"):
            assert abs(sum(lik for _, _, lik in source.lookup(origin)) - 1.0) < 1e-10

    def test_metadata_code_tables_shared_across_files(self, tmp_path):
        """Codes are assigned per source, not per file; a value first seen in
        the second file must still decode back to itself."""
        first = write_od_csv(tmp_path / "first.csv", ["A"], ["D1"], [1.0],
                             extra_columns={"mode": ["car"]})
        second = write_od_csv(tmp_path / "second.csv", ["B"], ["D2"], [1.0],
                              extra_columns={"mode": ["bus"]})
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config([first, second], metadata_columns={"mode": "mode"})
        )
        source.load_data()
        assert source.lookup("A")[0][1]["mode"] == "car"
        assert source.lookup("B")[0][1]["mode"] == "bus"

    def test_overlapping_origins_across_files_fail_loud(self, tmp_path):
        first = write_od_csv(tmp_path / "first.csv", ["A"], ["D1"], [1.0])
        second = write_od_csv(tmp_path / "second.csv", ["A"], ["D2"], [1.0])
        source = OriginDestinationMatrixSource("test_od", make_od_config([first, second]))
        with pytest.raises(Exception, match="disjoint"):
            source.load_data()

    def test_files_disagreeing_on_metadata_columns_fail_loud(self, tmp_path):
        first = write_od_csv(tmp_path / "first.csv", ["A"], ["D1"], [1.0],
                             extra_columns={"mode": ["car"], "code": ["X"]})
        second = write_od_csv(tmp_path / "second.csv", ["B"], ["D2"], [1.0],
                              extra_columns={"mode": ["bus"]})
        source = OriginDestinationMatrixSource(
            "test_od",
            make_od_config([first, second],
                           metadata_columns=[{"mode": "mode", "destination_code": "code"},
                                             {"mode": "mode"}]),
        )
        with pytest.raises(ValueError, match="metadata_columns"):
            source.load_data()

    def test_declared_metadata_column_absent_from_csv_fails_loud(self, tmp_path):
        csv_path = write_od_csv(tmp_path / "od.csv", ["A"], ["D1"], [1.0])
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, metadata_columns={"mode": "mode"})
        )
        with pytest.raises(Exception, match="mode"):
            source.load_data()


class TestOriginDestinationMatrixBoundaryPolicy:
    """Tests the out_of_boundary/on_empty/outside_value region-run policy,
    exercised via load_data(geo_units=...) + lookup() only."""

    def test_error_raises_when_out_of_boundary_destination_present(self, tmp_path):
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A", "A"], ["IN", "OUT"], [0.5, 0.5]
        )
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, out_of_boundary="error")
        )
        with pytest.raises(ValueError, match="out_of_boundary='error'"):
            source.load_data(geo_units={"A", "IN"})

    def test_redistribute_renormalizes_to_in_boundary_only(self, tmp_path):
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A", "A"], ["IN", "OUT"], [0.5, 0.5]
        )
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, out_of_boundary="redistribute", on_empty="error")
        )
        source.load_data(geo_units={"A", "IN"})
        result = source.lookup("A")
        assert len(result) == 1
        assert result[0][0] == "IN"
        assert abs(result[0][2] - 1.0) < 1e-10
        assert source._redistributed_fraction["A"] == 0.5

    def test_redistribute_on_empty_error_raises(self, tmp_path):
        csv_path = write_od_csv(tmp_path / "od.csv", ["A"], ["OUT"], [1.0])
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, out_of_boundary="redistribute", on_empty="error")
        )
        with pytest.raises(ValueError, match="on_empty='error'"):
            source.load_data(geo_units={"A"})

    def test_redistribute_on_empty_outside_uses_sentinel(self, tmp_path):
        csv_path = write_od_csv(tmp_path / "od.csv", ["A"], ["OUT"], [1.0])
        source = OriginDestinationMatrixSource(
            "test_od",
            make_od_config(csv_path, out_of_boundary="redistribute", on_empty="outside",
                            outside_value="OUTSIDE", sentinel_preserves=[]),
        )
        source.load_data(geo_units={"A"})
        result = source.lookup("A")
        assert len(result) == 1
        assert result[0][0] == "OUTSIDE"
        assert abs(result[0][2] - 1.0) < 1e-10

    def test_outside_policy_routes_out_of_boundary_mass_to_sentinel(self, tmp_path):
        # OUT1 and OUT2 share a mode, so they merge onto one sentinel row that
        # carries that mode rather than a stamped "OUTSIDE".
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A", "A", "A"], ["IN", "OUT1", "OUT2"], [0.5, 0.3, 0.2],
            extra_columns={"mode": ["car", "bus", "bus"]},
        )
        source = OriginDestinationMatrixSource(
            "test_od",
            make_od_config(csv_path, out_of_boundary="outside", outside_value="OUTSIDE",
                            metadata_columns={"mode": "mode"},
                            sentinel_preserves=["mode"]),
        )
        source.load_data(geo_units={"A", "IN"})
        result = source.lookup("A")
        by_dest = {d: (m, lik) for d, m, lik in result}
        assert set(by_dest) == {"IN", "OUTSIDE"}
        assert abs(by_dest["IN"][1] - 0.5) < 1e-10
        assert abs(by_dest["OUTSIDE"][1] - 0.5) < 1e-10
        assert by_dest["OUTSIDE"][0]["mode"] == "bus"

    def test_no_geo_units_is_a_no_op(self, tmp_path):
        csv_path = write_od_csv(tmp_path / "od.csv", ["A"], ["OUT"], [1.0])
        source = OriginDestinationMatrixSource(
            "test_od", make_od_config(csv_path, out_of_boundary="error")
        )
        source.load_data(geo_units=None)
        assert source.lookup("A") == [("OUT", {}, 1.0)]


class TestOriginDestinationSentinelPreserves:
    """`sentinel_preserves` names the metadata that survives the boundary.

    It is the grouping key for out-of-boundary rows, so it also bounds how many
    sentinel rows an origin can have: one per distinct combination of the named
    columns, not one per out-of-boundary row. Columns not named are dropped
    from the sentinel, never filled with a placeholder."""

    def _write(self, tmp_path):
        return write_od_csv(
            tmp_path / "od.csv", ["A"] * 4, ["IN", "OUT1", "OUT2", "OUT3"],
            [0.4, 0.2, 0.2, 0.2],
            extra_columns={"mode": ["car", "bus", "bus", "car"],
                            "dest_code": ["C0", "C1", "C2", "C3"]},
        )

    def _source(self, csv_path, sentinel_preserves, **kwargs):
        return OriginDestinationMatrixSource(
            "test_od",
            make_od_config(csv_path, out_of_boundary="outside", outside_value="OUTSIDE",
                            metadata_columns={"mode": "mode", "dest_code": "dest_code"},
                            sentinel_preserves=sentinel_preserves, **kwargs),
        )

    def test_required_when_outside_can_occur(self, tmp_path):
        with pytest.raises(ValueError, match="sentinel_preserves"):
            OriginDestinationMatrixSource(
                "test_od", make_od_config(tmp_path / "od.csv", out_of_boundary="outside",
                                           outside_value="OUTSIDE")
            )

    def test_required_under_redistribute_on_empty_outside(self, tmp_path):
        with pytest.raises(ValueError, match="sentinel_preserves"):
            OriginDestinationMatrixSource(
                "test_od",
                make_od_config(tmp_path / "od.csv", out_of_boundary="redistribute",
                                on_empty="outside", outside_value="OUTSIDE"),
            )

    def test_not_required_when_outside_cannot_occur(self, tmp_path):
        csv_path = write_od_csv(tmp_path / "od.csv", ["A"], ["IN"], [1.0])
        source = OriginDestinationMatrixSource("test_od", make_od_config(csv_path))
        source.load_data(geo_units={"A", "IN"})
        assert source.lookup("A") == [("IN", {}, 1.0)]

    def test_unknown_column_fails_loud(self, tmp_path):
        with pytest.raises(ValueError, match="not a declared metadata column"):
            self._source(tmp_path / "od.csv", ["mode", "not_a_column"])

    def test_sentinel_groups_only_on_preserved_columns(self, tmp_path):
        # dest_code is unique per destination, so grouping on it too would give
        # three sentinel rows instead of two.
        source = self._source(self._write(tmp_path), ["mode"])
        source.load_data(geo_units={"A", "IN"})
        outside = [(m["mode"], lik) for d, m, lik in source.lookup("A") if d == "OUTSIDE"]
        assert dict(outside) == {"bus": pytest.approx(0.4), "car": pytest.approx(0.2)}

    def test_non_preserved_metadata_is_omitted_not_filled(self, tmp_path):
        source = self._source(self._write(tmp_path), ["mode"])
        source.load_data(geo_units={"A", "IN"})
        rows = {d: m for d, m, _ in source.lookup("A")}
        assert set(rows["OUTSIDE"]) == {"mode"}          # dest_code dropped
        assert rows["IN"] == {"mode": "car", "dest_code": "C0"}

    def test_empty_list_gives_one_sentinel_row_per_origin(self, tmp_path):
        source = self._source(self._write(tmp_path), [])
        source.load_data(geo_units={"A", "IN"})
        outside = [r for r in source.lookup("A") if r[0] == "OUTSIDE"]
        assert len(outside) == 1
        assert outside[0][1] == {}
        assert outside[0][2] == pytest.approx(0.6)


class TestOriginDestinationNoValidDestinations:
    """An origin whose every destination is out-of-boundary takes the same
    grouped collapse as the `outside` policy.

    Its rows are real rows with real metadata, so there is nothing to invent:
    the sentinel carries the preserved metadata of the flow that left, one row
    per distinct combination, exactly as a partly-out-of-boundary origin does.
    """

    def _source(self, csv_path, sentinel_preserves, metadata_columns=None):
        return OriginDestinationMatrixSource(
            "test_od",
            make_od_config(csv_path, out_of_boundary="redistribute", on_empty="outside",
                            outside_value="OUTSIDE", metadata_columns=metadata_columns,
                            sentinel_preserves=sentinel_preserves),
        )

    def test_one_sentinel_row_per_preserved_combination(self, tmp_path):
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A"] * 3, ["OUT1", "OUT2", "OUT3"], [0.5, 0.3, 0.2],
            extra_columns={"mode": ["car", "bus", "bus"]},
        )
        source = self._source(csv_path, ["mode"], metadata_columns={"mode": "mode"})
        source.load_data(geo_units={"A"})
        outside = {m["mode"]: lik for d, m, lik in source.lookup("A") if d == "OUTSIDE"}
        assert outside == {"car": pytest.approx(0.5), "bus": pytest.approx(0.5)}

    def test_sentinel_rows_sum_to_one(self, tmp_path):
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A"] * 3, ["OUT1", "OUT2", "OUT3"], [3.0, 1.0, 6.0],
            extra_columns={"mode": ["car", "bus", "walk"]},
        )
        source = self._source(csv_path, ["mode"], metadata_columns={"mode": "mode"})
        source.load_data(geo_units={"A"})
        result = source.lookup("A")
        assert {d for d, _, _ in result} == {"OUTSIDE"}
        assert sum(lik for _, _, lik in result) == pytest.approx(1.0)

    def test_no_metadata_key_reports_the_sentinel_value(self, tmp_path):
        """Non-preserved columns are omitted from a sentinel row, never
        reported as `outside_value` — that names a destination and nothing
        else."""
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A"], ["OUT1"], [1.0],
            extra_columns={"mode": ["car"], "dest_code": ["C1"]},
        )
        source = self._source(csv_path, ["mode"],
                              metadata_columns={"mode": "mode", "dest_code": "dest_code"})
        source.load_data(geo_units={"A"})
        (_, metadata, _), = source.lookup("A")
        assert metadata == {"mode": "car"}


class TestOriginDestinationMasslessOrigin:
    """An origin whose rows all carry zero likelihood has nothing to normalise.

    A property of the data, not of the run — unrelated to the boundary. The
    observed metadata combinations are still real, so the origin is reported
    and weighted uniformly rather than dropped or failed on.
    """

    def test_uniform_weights_and_warning(self, tmp_path, caplog):
        import logging
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A", "A", "B"], ["IN1", "IN2", "IN1"], [0.0, 0.0, 1.0],
        )
        source = OriginDestinationMatrixSource("test_od", make_od_config(csv_path))
        with caplog.at_level(logging.WARNING):
            source.load_data(geo_units={"A", "B", "IN1", "IN2"})
        assert [lik for _, _, lik in source.lookup("A")] == [pytest.approx(0.5)] * 2
        assert [lik for _, _, lik in source.lookup("B")] == [pytest.approx(1.0)]
        assert any("massless" in message and "A" in message for message in caplog.messages)

    def test_massless_origin_outside_the_boundary_keeps_its_modes(self, tmp_path):
        """Masslessness is settled at load, so the sentinel collapse sees
        ordinary weights and the modes present in the data survive."""
        csv_path = write_od_csv(
            tmp_path / "od.csv", ["A"] * 3, ["OUT1", "OUT2", "OUT3"], [0.0, 0.0, 0.0],
            extra_columns={"mode": ["car", "bus", "bus"]},
        )
        source = OriginDestinationMatrixSource(
            "test_od",
            make_od_config(csv_path, out_of_boundary="redistribute", on_empty="outside",
                            outside_value="OUTSIDE", metadata_columns={"mode": "mode"},
                            sentinel_preserves=["mode"]),
        )
        source.load_data(geo_units={"A"})
        outside = {m["mode"]: lik for d, m, lik in source.lookup("A") if d == "OUTSIDE"}
        assert outside == {"car": pytest.approx(1 / 3), "bus": pytest.approx(2 / 3)}


class TestOriginDestinationBoundaryLevel:
    """Boundary membership is scoped to the declared `destination_level`.

    Census O-D tables list whole-country catch-alls ("Wales") in the same
    destination column as real LADs. Both are loaded geo unit names in a
    multi-nation run, just at different levels, so an unscoped check calls the
    catch-all in-boundary and hands a non-existent LAD downstream.
    """

    GEO_BY_LEVEL = {"LGU": {"Cardiff", "Glasgow City"}, "XLGU": {"Wales", "Scotland"}}
    GEO_FLAT = {"Cardiff", "Glasgow City", "Wales", "Scotland"}

    def _load(self, tmp_path, origins, destinations, likelihoods, *,
              destination_level="LGU", geo_units_by_level=None, geo_units=None,
              modes=None):
        """Build a one-file O-D source under out_of_boundary='outside' and load it."""
        csv_path = write_od_csv(
            tmp_path / "od.csv", origins, destinations, likelihoods,
            extra_columns={"work_mode": modes} if modes else None,
        )
        source = OriginDestinationMatrixSource(
            "test_od",
            make_od_config(csv_path, out_of_boundary="outside", outside_value="OUTSIDE",
                            destination_level=destination_level,
                            metadata_columns={"work_mode": "work_mode"} if modes else None,
                            sentinel_preserves=["work_mode"] if modes else []),
        )
        source.geo_units_by_level = geo_units_by_level
        source.load_data(geo_units=self.GEO_FLAT if geo_units is None else geo_units)
        return source

    def test_country_name_at_wrong_level_is_out_of_boundary(self, tmp_path):
        source = self._load(
            tmp_path, ["Glasgow City", "Glasgow City"], ["Cardiff", "Wales"], [0.6, 0.4],
            geo_units_by_level=self.GEO_BY_LEVEL,
        )
        dests = {d for d, _, _ in source.lookup("Glasgow City")}
        assert "Cardiff" in dests           # a real LGU stays
        assert "Wales" not in dests         # the XLGU catch-all does not
        assert "OUTSIDE" in dests

    def test_out_of_boundary_mass_is_preserved_not_renormalised(self, tmp_path):
        # The 0.4 must survive as OUTSIDE. Dropping it would renormalise
        # Cardiff to 1.0 and silently move cross-border commuters in-country.
        source = self._load(
            tmp_path, ["Glasgow City", "Glasgow City"], ["Cardiff", "Wales"], [0.6, 0.4],
            geo_units_by_level=self.GEO_BY_LEVEL,
        )
        by_dest = {d: lik for d, _, lik in source.lookup("Glasgow City")}
        assert by_dest["Cardiff"] == pytest.approx(0.6)
        assert by_dest["OUTSIDE"] == pytest.approx(0.4)

    def test_falls_back_to_flat_set_without_a_level_breakdown(self, tmp_path):
        # Direct programmatic loads pass no level map; behaviour is unchanged.
        source = self._load(
            tmp_path, ["Glasgow City"], ["Wales"], [1.0], geo_units_by_level=None,
        )
        assert source.lookup("Glasgow City")[0][0] == "Wales"

    def test_unknown_destination_level_fails_loud(self, tmp_path):
        with pytest.raises(ValueError, match="not a level in the loaded geography"):
            self._load(
                tmp_path, ["Glasgow City"], ["Cardiff"], [1.0],
                destination_level="NOT_A_LEVEL", geo_units_by_level=self.GEO_BY_LEVEL,
            )

    def test_sentinel_keeps_metadata_so_a_conditioned_draw_still_sees_it(self, tmp_path):
        # The out-of-area mass must remain reachable under each work mode. If
        # every out-of-boundary row collapsed onto one sentinel stamped
        # "OUTSIDE", a draw restricted to travelling modes would find no
        # sentinel and quietly renormalise that mass onto in-world destinations,
        # turning out-of-area workers into local ones.
        source = self._load(
            tmp_path,
            ["Cardiff"] * 4,
            ["Cardiff", "Cardiff", "Leeds", "Leeds"],
            [0.2, 0.3, 0.4, 0.1],
            modes=["From_Home", "Normal", "Normal", "Hybrid"],
            geo_units_by_level={"LGU": {"Cardiff"}},
            geo_units={"Cardiff"},
        )
        rows = source.lookup("Cardiff")
        outside = [(m["work_mode"], lik) for d, m, lik in rows if d == "OUTSIDE"]
        assert dict(outside) == {"Normal": pytest.approx(0.4), "Hybrid": pytest.approx(0.1)}
        assert sum(lik for _, _, lik in rows) == pytest.approx(1.0)

    def test_sentinel_rows_merge_when_metadata_matches(self, tmp_path):
        source = self._load(
            tmp_path,
            ["Cardiff"] * 3,
            ["Cardiff", "Leeds", "York"],
            [0.4, 0.3, 0.3],
            modes=["Normal", "Normal", "Normal"],
            geo_units_by_level={"LGU": {"Cardiff"}},
            geo_units={"Cardiff"},
        )
        outside = [r for r in source.lookup("Cardiff") if r[0] == "OUTSIDE"]
        assert len(outside) == 1                       # Leeds and York merged
        assert outside[0][2] == pytest.approx(0.6)

    def test_files_disagreeing_on_level_fails_loud(self, tmp_path):
        first = write_od_csv(tmp_path / "a.csv", ["Glasgow City"], ["Cardiff"], [1.0])
        second = write_od_csv(tmp_path / "b.csv", ["Cardiff"], ["Cardiff"], [1.0])
        config = make_od_config(first, out_of_boundary="outside", outside_value="OUTSIDE",
                                destination_level="LGU", sentinel_preserves=[])
        config["files"].append(
            dict(config["files"][0], path=str(second), destination_level="MGU")
        )
        source = OriginDestinationMatrixSource("test_od", config)
        source.geo_units_by_level = self.GEO_BY_LEVEL
        with pytest.raises(ValueError, match="disagree on 'destination_level'"):
            source.load_data(geo_units=self.GEO_FLAT)


# GUSamplerSource Tests

class TestGUSamplerSource:
    """Tests geographical unit sampler loading and lookup."""

    def _make_source_with_data(self, lookup_data, parent_attribute="workplace_location"):
        source = GUSamplerSource("test_gu_sampler", {"files": []})
        source._lookup = lookup_data
        source._parent_attribute = parent_attribute
        source._data_loaded = True
        return source

    def _person_in(self, parent_gu, parent_attribute="workplace_location"):
        return MinimalPerson(properties={parent_attribute: parent_gu})

    def test_lookup_resolves_parent_from_person_attribute(self):
        source = self._make_source_with_data({
            "ParentGU_A": {"SGU_1": 0.6, "SGU_2": 0.4}
        })
        result = source.lookup(self._person_in("ParentGU_A"), None, None)
        assert result == {"SGU_1": 0.6, "SGU_2": 0.4}

    def test_lookup_missing_parent_raises(self):
        source = self._make_source_with_data({
            "ParentGU_A": {"SGU_1": 1.0}
        })
        with pytest.raises(KeyError, match="no child-GU distribution"):
            source.lookup(self._person_in("MISSING"), None, None)

    def test_lookup_no_parent_attribute_value_raises(self):
        """Person missing the configured parent attribute → hard error."""
        source = self._make_source_with_data({"ParentGU_A": {"SGU_1": 1.0}})
        with pytest.raises(KeyError, match="workplace_location"):
            source.lookup(MinimalPerson(properties={}), None, None)

    def test_lookup_before_data_loaded_raises(self):
        source = GUSamplerSource("test_gu_sampler", {"files": []})
        with pytest.raises(RuntimeError, match="Data not loaded"):
            source.lookup(self._person_in("ParentGU_A"), None, None)

    def test_weight_normalization(self):
        """Weights should be normalized to probabilities summing to 1.0."""
        source = GUSamplerSource("test_gu_sampler", {"files": []})
        # Manually populate like load_data would
        raw_weights = {"SGU_1": 300.0, "SGU_2": 200.0, "SGU_3": 500.0}
        normalized = source._normalize_probabilities(raw_weights)
        assert abs(normalized["SGU_1"] - 0.3) < 1e-10
        assert abs(normalized["SGU_2"] - 0.2) < 1e-10
        assert abs(normalized["SGU_3"] - 0.5) < 1e-10

    def test_zero_weight_excluded(self):
        """
        During load_data, GUs with weight 0 are excluded before normalization.
        This tests the effect: only positive-weight GUs appear.
        """
        source = self._make_source_with_data({
            "ParentGU_A": {"SGU_1": 0.5, "SGU_2": 0.5}
            # SGU_3 with weight 0 would not appear
        })
        result = source.lookup(self._person_in("ParentGU_A"), None, None)
        assert "SGU_3" not in result
        assert len(result) == 2


# MultiKeyLookupSource Tests

class TestMultiKeyLookupSource:
    """Tests multi-key CSV lookups with different resolution types."""

    def _make_source(self, lookup_dict=None, key_columns_config=None,
                     value_columns=None, fallback=None, assignment_config=None):
        config = {
            "files": [{
                "key_columns": key_columns_config or {},
                "value_columns": value_columns or {},
            }],
            "fallback": fallback or {},
        }
        source = MultiKeyLookupSource(
            "test_multikey", config,
            assignment_config or MinimalAssignmentConfig()
        )
        source._lookup_dict = lookup_dict or {}
        source._key_columns_config = key_columns_config or {}
        source._value_columns = value_columns or {}
        source._data_loaded = True
        return source

    def test_direct_lookup_hit(self):
        """Direct attribute lookup finds matching row."""
        source = self._make_source(
            lookup_dict={("M", 30): {"cvd": 0.05, "crd": 0.03}},
            key_columns_config={
                "sex": {"attribute": "sex", "type": "direct"},
                "age": {"attribute": "age", "type": "direct"},
            },
            value_columns={"cvd": "cvd_prob", "crd": "crd_prob"},
        )
        person = MinimalPerson(age=30, sex="M")
        result = source.lookup(person)
        assert "cvd" in result
        assert "crd" in result

    def test_direct_lookup_miss_raises(self):
        """No fallbacks: a key not present in the data is an error."""
        source = self._make_source(
            lookup_dict={("M", 30): {"cvd": 0.05}},
            key_columns_config={
                "sex": {"attribute": "sex", "type": "direct"},
                "age": {"attribute": "age", "type": "direct"},
            },
        )
        person = MinimalPerson(age=99, sex="F")  # not in lookup
        with pytest.raises(KeyError, match="no row for key"):
            source.lookup(person)

    def test_empty_lookup_dict_raises(self):
        source = self._make_source(
            lookup_dict={},
            key_columns_config={
                "sex": {"attribute": "sex", "type": "direct"},
            },
        )
        with pytest.raises(RuntimeError, match="no data loaded"):
            source.lookup(MinimalPerson())

    def test_missing_key_attribute_raises(self):
        """If the person lacks an attribute the key needs, raise."""
        source = self._make_source(
            lookup_dict={("M", "W"): {"cvd": 0.05}},
            key_columns_config={
                "sex": {"attribute": "sex", "type": "direct"},
                "ethnicity": {"attribute": "ethnicity", "type": "direct"},
            },
        )
        person = MinimalPerson(sex="M", properties={})  # no ethnicity
        with pytest.raises(KeyError, match="could not resolve key column"):
            source.lookup(person)

    def test_direct_lookup_uses_properties_first(self):
        """Properties dict is checked before direct attributes."""
        source = self._make_source(
            lookup_dict={("W",): {"cvd": 0.05}},
            key_columns_config={
                "eth": {"attribute": "ethnicity", "type": "direct"},
            },
        )
        person = MinimalPerson(properties={"ethnicity": "W"})
        result = source.lookup(person)
        assert "cvd" in result

    def test_lookup_result_is_normalized(self):
        """Lookup results should be normalized to sum to 1."""
        source = self._make_source(
            lookup_dict={("M",): {"A": 3.0, "B": 7.0}},
            key_columns_config={
                "sex": {"attribute": "sex", "type": "direct"},
            },
        )
        person = MinimalPerson(sex="M")
        result = source.lookup(person)
        assert abs(sum(result.values()) - 1.0) < 1e-10
        assert abs(result["A"] - 0.3) < 1e-10

    def test_lookup_cache_works(self):
        """Same person key should hit cache on second lookup."""
        source = self._make_source(
            lookup_dict={("M", 30): {"cvd": 0.05}},
            key_columns_config={
                "sex": {"attribute": "sex", "type": "direct"},
                "age": {"attribute": "age", "type": "direct"},
            },
        )
        person = MinimalPerson(age=30, sex="M")
        result1 = source.lookup(person)
        result2 = source.lookup(person)
        assert result1 is result2  # same cached object

    def test_ancestor_lookup_resolution(self):
        """ancestor_lookup should traverse geo hierarchy."""
        lgu = MinimalGeoUnit("Manchester", level="LGU")
        sgu = MinimalGeoUnit("SGU_123", level="SGU", parent=lgu)

        source = self._make_source(
            lookup_dict={("Manchester",): {"industry_a": 0.5, "industry_b": 0.5}},
            key_columns_config={
                "region": {
                    "attribute": "geographical_unit",
                    "type": "ancestor_lookup",
                    "level": "LGU",
                    "property": "name",
                },
            },
        )
        person = MinimalPerson(geographical_unit=sgu)
        result = source.lookup(person)
        assert "industry_a" in result

    def test_ancestor_lookup_missing_geo_unit_raises(self):
        """No geo unit to resolve the key → raise."""
        source = self._make_source(
            lookup_dict={("Manchester",): {"industry_a": 1.0}},
            key_columns_config={
                "region": {
                    "attribute": "geographical_unit",
                    "type": "ancestor_lookup",
                    "level": "LGU",
                    "property": "name",
                },
            },
        )
        person = MinimalPerson(geographical_unit=None)
        with pytest.raises(KeyError, match="could not resolve key column"):
            source.lookup(person)

    def test_ancestor_lookup_falls_back_to_household_geo(self):
        """If person has no geo_unit, try household's."""
        lgu = MinimalGeoUnit("Birmingham", level="LGU")

        source = self._make_source(
            lookup_dict={("Birmingham",): {"industry_a": 1.0}},
            key_columns_config={
                "region": {
                    "attribute": "geographical_unit",
                    "type": "ancestor_lookup",
                    "level": "LGU",
                    "property": "name",
                },
            },
        )
        person = MinimalPerson(geographical_unit=None)
        household = MinimalHousehold(geographical_unit=lgu)
        result = source.lookup(person, household=household)
        assert "industry_a" in result

    def test_ancestor_lookup_keys_on_resolved_name(self):
        """An ancestor_lookup keys on the resolved ancestor name."""
        lgu = MinimalGeoUnit("East of England", level="LGU")
        sgu = MinimalGeoUnit("SGU_1", level="SGU", parent=lgu)
        source = self._make_source(
            lookup_dict={("East of England",): {"cvd": 1.0}},
            key_columns_config={
                "region": {
                    "attribute": "geographical_unit",
                    "type": "ancestor_lookup",
                    "level": "LGU",
                    "property": "name",
                },
            },
        )
        person = MinimalPerson(geographical_unit=sgu)
        assert "cvd" in source.lookup(person)

    def test_direct_keys_on_person_value(self):
        """A direct lookup keys on the person's attribute value."""
        source = self._make_source(
            lookup_dict={("M",): {"cvd": 0.05}},
            key_columns_config={
                "sex_col": {"attribute": "sex", "type": "direct"},
            },
        )
        person = MinimalPerson(sex="M")
        assert "cvd" in source.lookup(person)


# DataSourceManager._initialize_sources() Tests

class TestDataSourceManagerRouting:
    """
    Tests the routing logic in _initialize_sources that decides
    which DataSource subclass to instantiate.
    """

    def _make_config(self, data_sources):
        """Build a minimal config with data_sources dict."""
        config = MinimalAssignmentConfig()
        config.data_sources = {}
        for name, (source_type, source_config) in data_sources.items():
            config.data_sources[name] = MinimalDataSourceConfig(source_type, source_config)
        return config

    def test_format_geo_distribution(self):
        config = self._make_config({
            "ethnicity_distribution": ("csv_lookup", {
                "format": "geo_distribution",
                "files": [{"path": "/fake/path.csv", "key_columns": {"geo_unit": None}, "value_columns": {"W": "white"}}],
            })
        })
        manager = DataSourceManager(config)
        assert isinstance(manager.sources["ethnicity_distribution"], GeoDistributionSource)

    def test_format_diversity(self):
        config = self._make_config({
            "ethnicity_diversity": ("csv_lookup", {
                "format": "diversity",
                "files": [{"path": "/fake/path.csv", "key_columns": {"geo_unit": None}, "value_columns": {"single": "single"}}],
            })
        })
        manager = DataSourceManager(config)
        assert isinstance(manager.sources["ethnicity_diversity"], DiversitySource)

    def test_format_pair(self):
        config = self._make_config({
            "ethnicity_pairs": ("csv_lookup", {
                "format": "pair",
                "files": [{"path": "/fake/path.csv", "key_columns": {"geo_unit": None, "first_eth": None}, "value_columns": {"W": "white"}}],
            })
        })
        manager = DataSourceManager(config)
        assert isinstance(manager.sources["ethnicity_pairs"], PairProbabilitySource)

    def test_format_od_matrix(self):
        config = self._make_config({
            "commuting_flows": ("csv_lookup", {
                "format": "origin_destination_matrix",
                # out_of_boundary must be explicit. There is no default and silence is never interpreted.
                "out_of_boundary": "error",
                "files": [{"path": "/fake/path.csv", "key_columns": {"origin": "origin_col"}}],
            })
        })
        manager = DataSourceManager(config)
        assert isinstance(manager.sources["commuting_flows"], OriginDestinationMatrixSource)

    def test_format_multi_key(self):
        config = self._make_config({
            "disease_probs": ("csv_lookup", {
                "format": "multi_key",
                "files": [{
                    "path": "/fake/path.csv",
                    "key_columns": {
                        "sex": {"attribute": "sex", "type": "direct"},
                        "age": {"attribute": "age", "type": "category_lookup"},
                    },
                    "value_columns": {"cvd": "cvd_prob"},
                }],
            })
        })
        manager = DataSourceManager(config)
        assert isinstance(manager.sources["disease_probs"], MultiKeyLookupSource)

    def test_format_gu_sampler(self):
        config = self._make_config({
            "workplace_distribution": ("csv_lookup", {
                "format": "gu_sampler",
                "files": [{
                    "path": "/fake/path.csv",
                    "key_columns": {"LGU": None},
                    "geographical_unit_column": {"name": "SGU", "level": "SGU"},
                    "weight_column": "Total",
                }],
            })
        })
        manager = DataSourceManager(config)
        assert isinstance(manager.sources["workplace_distribution"], GUSamplerSource)

    def test_missing_format_raises(self):
        config = self._make_config({
            "no_format": ("csv_lookup", {
                "files": [{"path": "/fake/path.csv", "value_columns": {"W": "white"}}],
            })
        })
        with pytest.raises(ValueError, match="needs an explicit 'format'"):
            DataSourceManager(config)

    def test_constant_source_skipped(self):
        config = self._make_config({
            "fallback_constant": ("constant", {}),
        })
        manager = DataSourceManager(config)
        assert "fallback_constant" not in manager.sources

    def test_unknown_source_type_raises(self):
        config = self._make_config({
            "mystery": ("weird_type", {}),
        })
        with pytest.raises(ValueError, match="unknown type"):
            DataSourceManager(config)

    def test_get_source_returns_none_for_missing(self):
        config = self._make_config({})
        manager = DataSourceManager(config)
        assert manager.get_source("nonexistent") is None

    def test_lookup_delegates_to_source(self):
        config = self._make_config({
            "test_geo": ("csv_lookup", {
                "format": "geo_distribution",
                "files": [{"path": "/fake/path.csv", "value_columns": {"W": "white"}}],
            })
        })
        manager = DataSourceManager(config)
        # Manually set data on the source
        source = manager.sources["test_geo"]
        source._lookup = {"E00001": {"W": 1.0}}
        source._data_loaded = True

        person = MinimalPerson(geographical_unit=MinimalGeoUnit("E00001"))
        result = manager.lookup("test_geo", person, None, None)
        assert result == {"W": 1.0}

    def test_lookup_missing_source_raises(self):
        config = self._make_config({})
        manager = DataSourceManager(config)
        with pytest.raises(KeyError, match="not registered"):
            manager.lookup("nonexistent", "key")


# DiversitySource Tests

class TestDiversitySource:

    def _make_source_with_data(self, lookup_data, fallback=None):
        source = DiversitySource("test_diversity", {
            "files": [],
            "fallback": fallback or {},
        })
        source._lookup = lookup_data
        source._data_loaded = True
        return source

    def test_lookup_existing_geo_unit(self):
        source = self._make_source_with_data({
            "E00001": {"single": 0.7, "two": 0.2, "three_plus": 0.1}
        })
        result = source.lookup("E00001")
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_lookup_missing_raises(self):
        """No fallbacks: a missing geo unit is a hard error."""
        source = self._make_source_with_data({})
        with pytest.raises(KeyError, match="no diversity row"):
            source.lookup("MISSING")

    def test_lookup_before_data_loaded_raises(self):
        source = DiversitySource("test", {"files": []})
        with pytest.raises(RuntimeError, match="Data not loaded"):
            source.lookup("E00001")


# _require_declared_columns: a declared column absent from the file is fatal

class TestRequireDeclaredColumns:
    """The helper itself: what it reports, and that it reports it all at once."""

    @staticmethod
    def _frame():
        return pd.DataFrame({"geo_unit": ["E00001"], "white": [80.0]})

    def test_all_present_does_not_raise(self):
        _require_declared_columns(
            self._frame(), "test_geo", Path("/x/y.csv"),
            key_columns=["geo_unit"], value_columns={"W": "white"},
        )

    def test_reports_every_missing_column_at_once(self):
        """One raise names all of them, so a renamed file is fixed in one pass."""
        with pytest.raises(MissingDeclaredColumns) as excinfo:
            _require_declared_columns(
                self._frame(), "test_geo", Path("/x/y.csv"),
                key_columns=["geo_unit"],
                value_columns={"W": "white", "A": "asian", "B": "black"},
            )
        message = str(excinfo.value)
        assert "asian" in message and "black" in message

    def test_message_names_source_file_and_actual_header(self):
        with pytest.raises(MissingDeclaredColumns) as excinfo:
            _require_declared_columns(
                self._frame(), "test_geo", Path("/x/y.csv"),
                key_columns=["geo_unit"], value_columns={"A": "asian"},
            )
        message = str(excinfo.value)
        assert "test_geo" in message
        assert "/x/y.csv" in message
        assert "white" in message  # the columns the file does have

    def test_missing_key_column_is_reported(self):
        """A missing key column silently skips the geo filter, so it counts too."""
        with pytest.raises(MissingDeclaredColumns) as excinfo:
            _require_declared_columns(
                self._frame(), "test_geo", Path("/x/y.csv"),
                key_columns=["area_code"], value_columns={"W": "white"},
            )
        assert "area_code" in str(excinfo.value)


class TestDeclaredColumnsCheckedAtLoad:
    """Every loader checks its file before parsing, and the error is not wrapped.

    `MissingDeclaredColumns` must escape the per-file `except Exception` that
    turns parse errors into `RuntimeError`: a declared column that is not in
    the file is a config-contract violation, not a failure to load a file.
    """

    def test_geo_distribution_missing_value_column(self, tmp_path):
        path = tmp_path / "geo.csv"
        path.write_text("geo_unit,W\nE00001,8\n")
        source = GeoDistributionSource("test_geo", {"files": [{
            "path": str(path),
            "key_columns": {"geo_unit": None},
            "value_columns": {"W": "W", "A": "A"},
        }]})
        with pytest.raises(MissingDeclaredColumns, match="A"):
            source.load_data()

    def test_diversity_missing_value_column(self, tmp_path):
        path = tmp_path / "diversity.csv"
        path.write_text("geo_unit,single\nE00001,7\n")
        source = DiversitySource("test_diversity", {"files": [{
            "path": str(path),
            "key_columns": {"geo_unit": None},
            "value_columns": {"single": "single", "two": "two"},
        }]})
        with pytest.raises(MissingDeclaredColumns, match="two"):
            source.load_data()

    def test_pair_missing_value_column(self, tmp_path):
        path = tmp_path / "pair.csv"
        path.write_text("geo_unit,first_eth,prob_W\nE00001,W,0.9\n")
        source = PairProbabilitySource("test_pair", {"files": [{
            "path": str(path),
            "key_columns": {"geo_unit": None, "first_eth": None},
            "value_columns": {"W": "prob_W", "A": "prob_A"},
        }]})
        with pytest.raises(MissingDeclaredColumns, match="prob_A"):
            source.load_data()

    def test_multi_key_missing_value_column(self, tmp_path):
        """Already failed, but as a bare KeyError wrapped into RuntimeError."""
        path = tmp_path / "multi.csv"
        path.write_text("sex,age,cvd_prob\nm,30,0.1\n")
        source = MultiKeyLookupSource("test_multi", {"files": [{
            "path": str(path),
            "key_columns": {"sex": {"attribute": "sex", "type": "direct"},
                            "age": {"attribute": "age", "type": "direct"}},
            "value_columns": {"cvd": "cvd_prob", "crd": "crd_prob"},
        }]}, assignment_config=None)
        with pytest.raises(MissingDeclaredColumns, match="crd_prob"):
            source.load_data()

    def test_geo_distribution_missing_key_column(self, tmp_path):
        """Today this skips the geo filter, then dies at row[key_column]."""
        path = tmp_path / "geo.csv"
        path.write_text("area_code,W,A\nE00001,8,2\n")
        source = GeoDistributionSource("test_geo", {"files": [{
            "path": str(path),
            "key_columns": {"geo_unit": None},
            "value_columns": {"W": "W", "A": "A"},
        }]})
        with pytest.raises(MissingDeclaredColumns, match="geo_unit"):
            source.load_data()
