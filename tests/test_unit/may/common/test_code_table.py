import pandas as pd

from may.common.code_table import CodeTable


def test_encode_series_assigns_dense_first_appearance_codes():
    table = CodeTable()
    codes = table.encode_series(pd.Series(["b", "a", "b", "c"]))
    assert list(codes) == [0, 1, 0, 2]


def test_encode_series_stable_across_chunks():
    table = CodeTable()
    first = table.encode_series(pd.Series(["a", "b"]))
    second = table.encode_series(pd.Series(["b", "c", "a"]))
    assert list(first) == [0, 1]
    assert list(second) == [1, 2, 0]


def test_code_of_and_getitem_round_trip():
    table = CodeTable()
    table.encode_series(pd.Series(["x", "y", "x"]))
    assert table.code_of("x") == 0
    assert table.code_of("y") == 1
    assert table.code_of("missing") is None
    assert table[0] == "x"
    assert table[1] == "y"


def test_len_reflects_vocabulary_size_not_row_count():
    table = CodeTable()
    table.encode_series(pd.Series(["a", "a", "a", "b"]))
    assert len(table) == 2
