"""RouteLegsIndex: chunked, memory-bounded, columnar legs lookup (see
tmp/prd/route-legs-index-memory-reduction.md).

Tests assert external behaviour — which legs a lookup returns, with what
field values, and what errors bad inputs raise — never the internal
array/code-table layout.
"""

import pytest

from may.venue_distributor.route_legs_index import RouteLegsIndex


def write_legs_csv(path, rows):
    header = "origin_mgu,dest_mgu,mode_class,leg_idx,line_id,board_mgu,alight_mgu,t_board_min,t_alight_min\n"
    lines = [header]
    for row in rows:
        lines.append(",".join(str(v) for v in row) + "\n")
    path.write_text("".join(lines))
    return path


class TestModeClassFiltering:

    def test_filters_to_requested_class(self, tmp_path):
        csv_path = write_legs_csv(tmp_path / "legs.csv", [
            ("E1", "E2", "train", 0, "line_a", "E1", "E2", 0, 10),
            ("E1", "E2", "tube", 0, "line_b", "E1", "E2", 0, 5),
        ])
        index = RouteLegsIndex(legs_table_path=str(csv_path), class_filter="train")
        legs = index.get("E1", "E2", "train")
        assert len(legs) == 1
        assert legs[0]["line_id"] == "line_a"

    def test_other_class_absent_from_index(self, tmp_path):
        csv_path = write_legs_csv(tmp_path / "legs.csv", [
            ("E1", "E2", "train", 0, "line_a", "E1", "E2", 0, 10),
            ("E1", "E2", "tube", 0, "line_b", "E1", "E2", 0, 5),
        ])
        index = RouteLegsIndex(legs_table_path=str(csv_path), class_filter="train")
        assert index.get("E1", "E2", "tube") is None


class TestFieldRoundTrip:

    def test_metadata_and_structural_fields_preserved(self, tmp_path):
        csv_path = write_legs_csv(tmp_path / "legs.csv", [
            ("E1", "E2", "bus", 3, "line_a", "E5", "E6", 12, 19),
        ])
        index = RouteLegsIndex(
            legs_table_path=str(csv_path), class_filter="bus",
            leg_metadata={"t_board_min": "t_board_min", "t_alight_min": "t_alight_min"},
        )
        [leg] = index.get("E1", "E2", "bus")
        assert leg["line_id"] == "line_a"
        assert leg["board_mgu"] == "E5"
        assert leg["alight_mgu"] == "E6"
        assert leg["leg_idx"] == 3
        assert leg["t_board_min"] == 12
        assert leg["t_alight_min"] == 19


class TestColumnValidation:

    def test_missing_required_column_raises_at_construction(self, tmp_path):
        csv_path = tmp_path / "legs.csv"
        csv_path.write_text("origin_mgu,dest_mgu,mode_class,leg_idx,line_id,board_mgu\n"
                             "E1,E2,bus,0,line_a,E1\n")
        with pytest.raises(ValueError, match="alight_mgu"):
            RouteLegsIndex(legs_table_path=str(csv_path), class_filter="bus")

    def test_missing_config_metadata_column_raises_at_construction(self, tmp_path):
        csv_path = write_legs_csv(tmp_path / "legs.csv", [
            ("E1", "E2", "bus", 0, "line_a", "E1", "E2", 0, 10),
        ])
        with pytest.raises(ValueError, match="fare_pence"):
            RouteLegsIndex(
                legs_table_path=str(csv_path), class_filter="bus",
                leg_metadata={"fare": "fare_pence"},
            )


class TestAbsentFile:

    def test_absent_file_yields_empty_index(self, tmp_path):
        index = RouteLegsIndex(legs_table_path=str(tmp_path / "missing.csv"), class_filter="bus")
        assert index.get("E1", "E2", "bus") is None


class TestNoFilter:

    def test_no_filter_indexes_all_classes(self, tmp_path):
        csv_path = write_legs_csv(tmp_path / "legs.csv", [
            ("E1", "E2", "train", 0, "line_a", "E1", "E2", 0, 10),
            ("E1", "E2", "tube", 0, "line_b", "E1", "E2", 0, 5),
        ])
        index = RouteLegsIndex(legs_table_path=str(csv_path), class_filter=None)
        assert index.get("E1", "E2", "train")[0]["line_id"] == "line_a"
        assert index.get("E1", "E2", "tube")[0]["line_id"] == "line_b"

    def test_key_collision_across_classes_does_not_merge(self, tmp_path):
        """Regression: (origin, dest) pairs can collide across mode_class in
        the real national table (verified against route_legs.csv). Dropping
        mode_class from the key would silently merge these."""
        csv_path = write_legs_csv(tmp_path / "legs.csv", [
            ("E1", "E2", "train", 0, "line_a", "E1", "E2", 0, 10),
            ("E1", "E2", "tube", 0, "line_b", "E1", "E2", 0, 5),
        ])
        index = RouteLegsIndex(legs_table_path=str(csv_path), class_filter=None)
        train_legs = index.get("E1", "E2", "train")
        tube_legs = index.get("E1", "E2", "tube")
        assert len(train_legs) == 1 and len(tube_legs) == 1
        assert train_legs[0]["line_id"] == "line_a"
        assert tube_legs[0]["line_id"] == "line_b"


class TestChunking:

    def test_fixture_smaller_than_one_chunk(self, tmp_path):
        csv_path = write_legs_csv(tmp_path / "legs.csv", [
            ("E1", "E2", "bus", 0, "line_a", "E1", "E2", 0, 10),
        ])
        index = RouteLegsIndex(legs_table_path=str(csv_path), class_filter="bus", chunk_size=500_000)
        assert index.get("E1", "E2", "bus")[0]["line_id"] == "line_a"

    def test_fixture_spanning_multiple_chunks(self, tmp_path):
        rows = []
        for i in range(10):
            rows.append((f"O{i}", f"D{i}", "bus", 0, f"line_{i}", f"O{i}", f"D{i}", 0, 10))
        csv_path = write_legs_csv(tmp_path / "legs.csv", rows)
        index = RouteLegsIndex(legs_table_path=str(csv_path), class_filter="bus", chunk_size=3)
        for i in range(10):
            legs = index.get(f"O{i}", f"D{i}", "bus")
            assert legs is not None and legs[0]["line_id"] == f"line_{i}"

    def test_multi_leg_journey_split_across_chunk_boundary(self, tmp_path):
        """Two legs of the same (origin, dest, mode_class) journey landing in
        different chunks must still end up in the same lookup result."""
        rows = [
            ("O0", "D0", "bus", 0, "line_first", "O0", "M0", 0, 5),
        ]
        for i in range(1, 6):
            rows.append((f"O{i}", f"D{i}", "bus", 0, f"line_{i}", f"O{i}", f"D{i}", 0, 10))
        rows.append(("O0", "D0", "bus", 1, "line_second", "M0", "D0", 5, 15))
        csv_path = write_legs_csv(tmp_path / "legs.csv", rows)
        index = RouteLegsIndex(legs_table_path=str(csv_path), class_filter="bus", chunk_size=3)
        legs = index.get("O0", "D0", "bus")
        assert {leg["line_id"] for leg in legs} == {"line_first", "line_second"}
