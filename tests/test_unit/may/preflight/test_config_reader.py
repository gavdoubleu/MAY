import os

import pytest

from may.preflight.config_reader import FAILED, ConfigReader
from may.preflight.report import Finding, PreflightReport

BLANK_OPTIONAL = "left blank; give a value, or remove the key to use the default"
QUOTE_HINT = "put it in quotes to use it as text"


def reader_over(value, key_path="block", data_dir=None):
    report = PreflightReport()
    return ConfigReader(value, key_path, report, data_dir=data_dir), report


def problems(report):
    return [(finding.key, finding.message) for finding in report.problems]


def warnings(report):
    return [(finding.key, finding.message) for finding in report.warnings]


# --- FAILED ---------------------------------------------------------------------------


def test_failed_repr_and_bool_raises():
    assert repr(FAILED) == "FAILED"
    with pytest.raises(TypeError, match="is FAILED"):
        bool(FAILED)


# --- leaves: shared cases --------------------------------------------------------------

LEAF_CASES = [
    # (method, good value, expected read, wrong-type value, wrong-type message)
    ("text", "abc", "abc", [1], "must be text; got a list"),
    ("name", "abc", "abc", {"a": 1}, "must be text; got a mapping"),
    ("true_or_false", True, True, "yes", "must be true or false; got text 'yes'"),
    ("whole_number", 3, 3, 2.5, "must be a whole number; got a number 2.5"),
    ("number", 2.5, 2.5, "x", "must be a number; got text 'x'"),
    ("path", "some/file.csv", "some/file.csv", 2.5,
     f"must be text; got a number 2.5; {QUOTE_HINT}"),
]
LEAF_METHODS = [case[0] for case in LEAF_CASES]


@pytest.mark.parametrize("method, good, expected, _bad, _message", LEAF_CASES)
def test_leaf_good_value(method, good, expected, _bad, _message):
    reader, report = reader_over({"key": good})
    assert getattr(reader, method)("key") == expected
    assert problems(report) == []


@pytest.mark.parametrize("method", LEAF_METHODS)
def test_leaf_absent_required(method):
    reader, report = reader_over({})
    assert getattr(reader, method)("key", give="the thing") is FAILED
    assert problems(report) == [("block.key", "not set; give the thing")]


@pytest.mark.parametrize("method", LEAF_METHODS)
def test_leaf_absent_required_default_noun(method):
    reader, report = reader_over({})
    getattr(reader, method)("key")
    assert problems(report) == [("block.key", "not set; give a value")]


@pytest.mark.parametrize("method", LEAF_METHODS)
def test_leaf_absent_optional_takes_default_silently(method):
    reader, report = reader_over({})
    assert getattr(reader, method)("key", default="fallback") == "fallback"
    assert getattr(reader, method)("key", default=None) is None
    assert problems(report) == [] and warnings(report) == []


@pytest.mark.parametrize("method", LEAF_METHODS)
def test_leaf_blank_required(method):
    reader, report = reader_over({"key": None})
    assert getattr(reader, method)("key", give="the thing") is FAILED
    assert problems(report) == [("block.key", "left blank; give the thing")]


@pytest.mark.parametrize("method", LEAF_METHODS)
def test_leaf_blank_optional(method):
    reader, report = reader_over({"key": None})
    assert getattr(reader, method)("key", default=None) is FAILED
    assert problems(report) == [("block.key", BLANK_OPTIONAL)]


@pytest.mark.parametrize("method, _good, _expected, bad, message", LEAF_CASES)
def test_leaf_wrong_type(method, _good, _expected, bad, message):
    reader, report = reader_over({"key": bad})
    assert getattr(reader, method)("key") is FAILED
    assert problems(report) == [("block.key", message)]


# --- type rules ------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["text", "name", "path"])
def test_whole_number_accepted_as_text(method):
    reader, report = reader_over({"key": 2021})
    assert getattr(reader, method)("key") == "2021"
    assert problems(report) == []


def test_names_accept_whole_numbers_as_text():
    reader, report = reader_over({"key": [2021, "msoa"]})
    assert reader.names("key") == ["2021", "msoa"]
    assert problems(report) == []


@pytest.mark.parametrize(
    "value, message",
    [
        (2.5, f"must be text; got a number 2.5; {QUOTE_HINT}"),
        (True, f"must be text; got true; {QUOTE_HINT}"),
        (False, f"must be text; got false; {QUOTE_HINT}"),
    ],
)
def test_float_or_bool_as_text_gives_quoting_hint(value, message):
    reader, report = reader_over({"key": value})
    assert reader.text("key") is FAILED
    assert problems(report) == [("block.key", message)]


@pytest.mark.parametrize("method", ["whole_number", "number"])
def test_bool_is_not_a_number(method):
    reader, report = reader_over({"key": True})
    assert getattr(reader, method)("key") is FAILED
    assert problems(report)[0][1].endswith("; got true")


def test_whole_number_accepted_as_number():
    reader, report = reader_over({"key": 3})
    assert reader.number("key") == 3
    assert problems(report) == []


@pytest.mark.parametrize(
    "value, message",
    [
        ("true", "must be true or false; got text 'true'"),
        (1, "must be true or false; got a whole number 1"),
    ],
)
def test_text_or_number_is_not_true_or_false(value, message):
    reader, report = reader_over({"key": value})
    assert reader.true_or_false("key") is FAILED
    assert problems(report) == [("block.key", message)]


def test_scalar_not_wrapped_into_list():
    reader, report = reader_over({"key": "msoa"})
    assert reader.names("key") is FAILED
    assert problems(report) == [("block.key", "must be a list; got text 'msoa'")]


# --- name / names / one_of -------------------------------------------------------------


def test_name_empty_text_is_problem():
    reader, report = reader_over({"key": ""})
    assert reader.name("key", give="a venue type") is FAILED
    assert problems(report) == [("block.key", "must not be empty; give a venue type")]


def test_names_bad_item_left_out_siblings_kept():
    reader, report = reader_over({"key": ["a", None, ["x"], "", "b"]})
    assert reader.names("key") == ["a", "b"]
    assert problems(report) == [
        ("block.key[2]", "left blank; give a name"),
        ("block.key[3]", "must be text; got a list"),
        ("block.key[4]", "must not be empty; give a name"),
    ]


def test_names_blank_and_absent():
    reader, report = reader_over({"blank": None})
    assert reader.names("blank", give="level names") is FAILED
    assert reader.names("absent", default=[]) == []
    assert problems(report) == [("block.blank", "left blank; give level names")]


def test_one_of_good_and_sorted_choices_on_miss():
    reader, report = reader_over({"good": "b", "bad": "z"})
    assert reader.one_of("good", ["c", "a", "b"]) == "b"
    assert reader.one_of("bad", ["c", "a", "b"]) is FAILED
    assert problems(report) == [("block.bad", "must be one of a, b, c; got text 'z'")]


def test_one_of_absent_blank_and_whole_number():
    reader, report = reader_over({"blank": None, "number": 1})
    assert reader.one_of("absent", ["a"], default="a") == "a"
    assert reader.one_of("blank", ["a"], default="a") is FAILED
    assert reader.one_of("number", ["1", "2"]) == "1"
    assert problems(report) == [("block.blank", BLANK_OPTIONAL)]


def test_one_of_wrong_type():
    reader, report = reader_over({"key": [1]})
    assert reader.one_of("key", ["a", "b"]) is FAILED
    assert problems(report) == [("block.key", "must be one of a, b; got a list")]


# --- path ------------------------------------------------------------------------------


def test_path_resolves_roots(monkeypatch):
    from may.utils import path_resolver as pr

    monkeypatch.setattr(pr, "_resolver", pr.PathResolver("/cfg", "/data", "/out"))
    reader, _ = reader_over({"key": "${data_root}/file.csv"})
    assert reader.path("key") == "/data/file.csv"


# --- value display ---------------------------------------------------------------------


def test_long_value_truncated():
    long_text = "x" * 60
    reader, report = reader_over({"key": long_text})
    reader.whole_number("key")
    message = problems(report)[0][1]
    shown = message.split("got text ")[1]
    assert shown == repr(long_text)[:40] + "…"


def test_mapping_and_list_shown_by_term():
    reader, report = reader_over({"a": {"x": 1}, "b": [1, 2]})
    reader.number("a")
    reader.number("b")
    assert problems(report) == [
        ("block.a", "must be a number; got a mapping"),
        ("block.b", "must be a number; got a list"),
    ]


def test_null_shown_as_null():
    reader, report = reader_over({"key": [None]})
    reader.items("key")
    assert problems(report) == [("block.key[1]", "must be a mapping; got null")]


# --- value -----------------------------------------------------------------------------


def test_value_raw_records_nothing():
    reader, report = reader_over({"key": [1, "a"], "blank": None})
    assert reader.value("key") == [1, "a"]
    assert reader.value("blank") is None
    assert reader.value("absent") is None
    assert reader.value("absent", default=5) == 5
    assert reader.value() == {"key": [1, "a"], "blank": None}
    assert problems(report) == []


# --- section ---------------------------------------------------------------------------


def test_section_reads_nested():
    reader, report = reader_over({"inner": {"key": 3}})
    inner = reader.section("inner")
    assert inner.key_path == "block.inner"
    assert not inner.failed
    assert inner.whole_number("key") == 3
    assert problems(report) == []


@pytest.mark.parametrize(
    "block, message",
    [
        ({"inner": None}, "left blank; give its settings"),
        ({"inner": [1]}, "must be a mapping; got a list"),
        ({}, "not set; give its settings"),
    ],
)
def test_bad_section_is_inert(block, message):
    reader, report = reader_over(block)
    inner = reader.section("inner", give="its settings")
    assert inner.failed
    assert problems(report) == [("block.inner", message)]
    # every read on an inert reader is FAILED and records nothing
    assert inner.text("a") is FAILED
    assert inner.name("a") is FAILED
    assert inner.names("a") is FAILED
    assert inner.true_or_false("a") is FAILED
    assert inner.whole_number("a") is FAILED
    assert inner.number("a") is FAILED
    assert inner.one_of("a", ["x"]) is FAILED
    assert inner.path("a") is FAILED
    assert inner.data_file("a", "/tmp") is FAILED
    assert inner.value("a") is FAILED
    assert inner.value() is FAILED
    assert inner.keys() is FAILED
    assert inner.items("a") is FAILED
    assert inner.entries("a") is FAILED
    assert inner.section("a").failed
    assert inner.section("a").section("b").failed
    assert inner.config_file("a").failed
    assert len(report.problems) == 1


def test_blank_optional_section_is_problem():
    reader, report = reader_over({"inner": None})
    assert reader.section("inner", default={}).failed
    assert problems(report) == [("block.inner", BLANK_OPTIONAL)]


def test_absent_optional_section_is_empty_reader():
    reader, report = reader_over({})
    inner = reader.section("inner", default={})
    assert not inner.failed
    assert inner.keys() == []
    assert inner.whole_number("key", default=7) == 7
    assert problems(report) == []



@pytest.mark.parametrize("method", ["section", "items", "entries"])
def test_navigation_absent_default_none_is_none_silently(method):
    reader, report = reader_over({})
    assert getattr(reader, method)("key", default=None) is None
    assert problems(report) == []

# --- items / entries / keys ------------------------------------------------------------


def test_items_bad_item_left_out_siblings_read():
    reader, report = reader_over({"steps": [{"a": 1}, "oops", {"a": "x"}]})
    steps = reader.items("steps")
    assert [step.key_path for step in steps] == ["block.steps[1]", "block.steps[3]"]
    assert steps[0].whole_number("a") == 1
    assert steps[1].whole_number("a") is FAILED
    assert problems(report) == [
        ("block.steps[2]", "must be a mapping; got text 'oops'"),
        ("block.steps[3].a", "must be a whole number; got text 'x'"),
    ]


def test_items_blank_wrong_type_absent():
    reader, report = reader_over({"blank": None, "scalar": {"a": 1}})
    assert reader.items("blank") is FAILED
    assert reader.items("scalar") is FAILED
    assert reader.items("absent") is FAILED
    assert reader.items("optional", default=[]) == []
    assert problems(report) == [
        ("block.blank", "left blank; give a value"),
        ("block.scalar", "must be a list; got a mapping"),
        ("block.absent", "not set; give a value"),
    ]


def test_entries_by_name_int_name_as_text_bad_left_out():
    reader, report = reader_over(
        {"types": {"school": {"size": 3}, 2021: {"size": 4}, "bad": 5}}
    )
    entries = reader.entries("types")
    assert list(entries) == ["school", "2021"]
    assert entries["school"].key_path == "block.types.school"
    assert entries["2021"].key_path == "block.types.2021"
    assert entries["2021"].whole_number("size") == 4
    assert problems(report) == [("block.types.bad", "must be a mapping; got a whole number 5")]



def test_entries_names_same_as_text_is_problem_first_kept():
    reader, report = reader_over({"types": {1: {"size": 3}, "1": {"size": 4}}})
    entries = reader.entries("types")
    assert list(entries) == ["1"]
    assert entries["1"].whole_number("size") == 3
    assert problems(report) == [
        ("block.types.1", "named twice; give each entry a different name")
    ]


@pytest.mark.parametrize(
    "name, shown",
    [
        (True, f"true; {QUOTE_HINT}"),
        (2.5, f"a number 2.5; {QUOTE_HINT}"),
        (None, "null"),
    ],
)
def test_entries_name_not_text_is_problem_left_out(name, shown):
    reader, report = reader_over({"types": {name: {"size": 3}, "school": {"size": 4}}})
    entries = reader.entries("types")
    assert list(entries) == ["school"]
    assert problems(report) == [
        (f"block.types.{name}", f"entry name must be text; got {shown}")
    ]

def test_entries_blank_wrong_type_absent():
    reader, report = reader_over({"blank": None, "list": [1]})
    assert reader.entries("blank") is FAILED
    assert reader.entries("list") is FAILED
    assert reader.entries("absent", default={}) == {}
    assert problems(report) == [
        ("block.blank", "left blank; give a value"),
        ("block.list", "must be a mapping; got a list"),
    ]


def test_keys_as_written():
    reader, _ = reader_over({"msoa": "a.csv", "lsoa": "b.csv"})
    assert reader.keys() == ["msoa", "lsoa"]


# --- key paths -------------------------------------------------------------------------


def test_nested_key_path_one_based():
    reader, report = reader_over({"a": {"b": [{}, {"c": None}]}}, key_path="")
    item = reader.section("a").items("b")[1]
    item.text("c")
    assert problems(report) == [("a.b[2].c", "left blank; give a value")]


def test_root_with_empty_path():
    reader, report = reader_over({}, key_path="")
    reader.text("key")
    assert problems(report) == [("key", "not set; give a value")]


def test_root_reader_absolute_path_shares_report():
    reader, report = reader_over({}, key_path="timeline[1].config (x.yaml)")
    households = reader.root_reader("households", {"data_dir": None})
    assert households.key_path == "households"
    households.path("data_dir")
    assert problems(report) == [("households.data_dir", "left blank; give a value")]


@pytest.mark.parametrize(
    "value, message",
    [(None, "left blank; give a value"), ([1], "must be a mapping; got a list")],
)
def test_root_reader_over_bad_value_is_inert(value, message):
    reader, report = reader_over({})
    households = reader.root_reader("households", value)
    assert households.failed
    assert problems(report) == [("households", message)]


def test_root_reader_keeps_data_dir(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    reader, report = reader_over({}, data_dir=str(tmp_path))
    other = reader.root_reader("other", {"file": "a.yaml"})
    (tmp_path / "a.yaml").write_text("k: 1\n")
    assert other.config_file("file").whole_number("k") == 1
    assert problems(report) == []


# --- problem / warning -----------------------------------------------------------------


def test_problem_and_warning_at_self_child_and_item():
    reader, report = reader_over({})
    reader.problem("whole block bad")
    reader.problem("at least 1", at="max_contacts")
    reader.problem("bad item", at="[2]")
    reader.warning("heads up", at="x")
    assert problems(report) == [
        ("block", "whole block bad"),
        ("block.max_contacts", "at least 1"),
        ("block[2]", "bad item"),
    ]
    assert warnings(report) == [("block.x", "heads up")]


def test_problem_on_root_with_empty_path():
    reader, report = reader_over({}, key_path="")
    reader.problem("oops", at="key")
    assert problems(report) == [("key", "oops")]


def test_duplicate_problem_dropped():
    reader, report = reader_over({"key": None})
    reader.text("key")
    reader.text("key")
    assert len(report.problems) == 1


def test_one_report_shared_by_descendants():
    reader, report = reader_over({"a": {"b": [{"c": None}]}, "d": {"e": None}})
    reader.section("a").items("b")[0].text("c")
    reader.section("d").text("e")
    assert [key for key, _ in problems(report)] == ["block.a.b[1].c", "block.d.e"]


def test_problem_on_inert_reader_still_recorded():
    reader, report = reader_over({})
    inert = reader.section("missing")
    inert.problem("domain check", at="x")
    assert ("block.missing.x", "domain check") in problems(report)


def test_failed_only_for_inert_readers():
    reader, report = reader_over({"a": {}})
    inner = reader.section("a")
    inner.problem("domain problem")
    assert not inner.failed
    assert not reader.failed


# --- config_file -----------------------------------------------------------------------


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_config_file_success_reads_inner_keys(tmp_path):
    write(tmp_path / "step.yaml", "eligibility:\n  groups:\n    - {}\n    - filters: null\n")
    reader, report = reader_over({"config": "step.yaml"}, key_path="timeline[3]",
                                 data_dir=str(tmp_path))
    step = reader.config_file("config")
    found = os.path.join(str(tmp_path), "step.yaml")
    assert step.key_path == f"timeline[3].config ({found})"
    step.section("eligibility").items("groups")[1].value("filters")
    step.section("eligibility").items("groups")[1].names("filters")
    assert problems(report) == [
        (f"timeline[3].config ({found}): eligibility.groups[2].filters",
         "left blank; give a value"),
    ]


def test_config_file_reader_carries_found_path(tmp_path):
    write(tmp_path / "step.yaml", "a: 1\n")
    reader, _ = reader_over({"config": "step.yaml"}, data_dir=str(tmp_path))
    step = reader.config_file("config")
    assert step.found_path == os.path.join(str(tmp_path), "step.yaml")
    assert step.section("a", default={}).found_path is None
    assert reader.found_path is None


def test_config_file_failed_reader_has_no_found_path(tmp_path):
    reader, _ = reader_over({"config": "missing.yaml"}, data_dir=str(tmp_path))
    assert reader.config_file("config").found_path is None


def test_config_file_problem_directly_under_file(tmp_path):
    write(tmp_path / "step.yaml", "a: null\n")
    reader, report = reader_over({"config": str(tmp_path / "step.yaml")})
    step = reader.config_file("config")
    step.text("a")
    step.problem("file-level")
    found = str(tmp_path / "step.yaml")
    assert problems(report) == [
        (f"block.config ({found}): a", "left blank; give a value"),
        (f"block.config ({found})", "file-level"),
    ]


def test_config_file_explicit_data_dir(tmp_path):
    write(tmp_path / "s.yaml", "k: 1\n")
    reader, report = reader_over({"config": "s.yaml"})
    assert reader.config_file("config", data_dir=str(tmp_path)).whole_number("k") == 1
    assert problems(report) == []


def test_config_file_default_path(tmp_path):
    write(tmp_path / "s.yaml", "k: 1\n")
    reader, report = reader_over({}, data_dir=str(tmp_path))
    assert reader.config_file("config", default="s.yaml").whole_number("k") == 1
    assert problems(report) == []


@pytest.mark.parametrize(
    "contents, message",
    [
        (None, "not found; tried {path}"),
        ("a: [unclosed\n", "{path} is not valid YAML at line 2, column 1"),
        (b"\xff\xfe\x00bad", "{path} could not be read: not UTF-8 text"),
        ("", "{path} is empty"),
        ("- a\n- b\n", "{path} must hold a mapping; got a list"),
    ],
)
def test_config_file_failures_one_problem_inert(tmp_path, contents, message):
    path = tmp_path / "s.yaml"
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    elif contents is not None:
        write(path, contents)
    reader, report = reader_over({"config": str(path)})
    step = reader.config_file("config")
    assert step.failed
    assert step.text("anything") is FAILED
    assert problems(report) == [("block.config", message.format(path=str(path)))]


def test_config_file_bad_key(tmp_path):
    reader, report = reader_over({"blank": None, "list": [1]})
    assert reader.config_file("blank").failed
    assert reader.config_file("list").failed
    assert reader.config_file("absent").failed
    assert problems(report) == [
        ("block.blank", "left blank; give a value"),
        ("block.list", "must be text; got a list"),
        ("block.absent", "not set; give a value"),
    ]


def test_config_file_failed_data_dir_is_inert_silently(tmp_path):
    reader, report = reader_over({"config": "s.yaml"}, data_dir=FAILED)
    assert reader.config_file("config").failed
    assert problems(report) == []


# --- data_file -------------------------------------------------------------------------


def test_data_file_one_path(tmp_path):
    write(tmp_path / "a.csv", "x")
    reader, report = reader_over({"file": "a.csv"})
    assert reader.data_file("file", str(tmp_path)) == [str(tmp_path / "a.csv")]
    assert problems(report) == []


def test_data_file_list(tmp_path):
    write(tmp_path / "a.csv", "x")
    write(tmp_path / "b.csv", "x")
    reader, _ = reader_over({"file": ["a.csv", "b.csv"]})
    assert reader.data_file("file", str(tmp_path)) == [
        str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    ]


def test_data_file_missing(tmp_path):
    write(tmp_path / "a.csv", "x")
    reader, report = reader_over({"file": ["a.csv", "b.csv"]})
    assert reader.data_file("file", str(tmp_path)) is FAILED
    assert problems(report) == [("block.file", f"not found: {tmp_path / 'b.csv'}")]


@pytest.mark.parametrize(
    "spec, shown",
    [([], "a list"), (True, "true"), (["a.csv", 2.5], "a list"), ({"a": 1}, "a mapping")],
)
def test_data_file_bad_shape(tmp_path, spec, shown):
    reader, report = reader_over({"file": spec})
    assert reader.data_file("file", str(tmp_path)) is FAILED
    assert problems(report) == [
        ("block.file", f"must be a path or a non-empty list of paths; got {shown}")
    ]


def test_data_file_whole_numbers_read_as_text(tmp_path):
    write(tmp_path / "2021", "x")
    write(tmp_path / "7", "x")
    reader, report = reader_over({"one": 2021, "many": ["7", 2021]})
    assert reader.data_file("one", str(tmp_path)) == [str(tmp_path / "2021")]
    assert reader.data_file("many", str(tmp_path)) == [str(tmp_path / "7"), str(tmp_path / "2021")]
    assert problems(report) == []


def test_data_file_failed_data_dir_checks_shape_only(tmp_path):
    reader, report = reader_over({"good": "nowhere.csv", "bad": {"a": 1}})
    assert reader.data_file("good", FAILED) is FAILED
    assert reader.data_file("bad", FAILED) is FAILED
    assert problems(report) == [
        ("block.bad", "must be a path or a non-empty list of paths; got a mapping")
    ]


def test_data_file_blank_absent(tmp_path):
    reader, report = reader_over({"blank": None})
    assert reader.data_file("blank", str(tmp_path)) is FAILED
    assert reader.data_file("absent", str(tmp_path), default=None) is None
    assert reader.data_file("required", str(tmp_path), give="a data file") is FAILED
    assert problems(report) == [
        ("block.blank", "left blank; give a value"),
        ("block.required", "not set; give a data file"),
    ]


def test_data_file_default_is_found_and_checked(tmp_path):
    write(tmp_path / "a.csv", "x")
    reader, report = reader_over({})
    assert reader.data_file("found", str(tmp_path), default="a.csv") == [str(tmp_path / "a.csv")]
    assert reader.data_file("missing", str(tmp_path), default="b.csv") is FAILED
    assert problems(report) == [("block.missing", f"not found: {tmp_path / 'b.csv'}")]


# --- domain-free -----------------------------------------------------------------------


def test_reader_has_no_domain_imports():
    import ast
    import inspect

    from may.preflight import config_reader

    tree = ast.parse(inspect.getsource(config_reader))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    may_imports = {name for name in imported if name and name.startswith("may")}
    assert may_imports <= {
        "may.preflight.report", "may.utils", "may.utils.path_resolver", "may.utils.yaml_loader"
    }


def test_config_file_absent_default_none_is_none_silently(tmp_path):
    reader, report = reader_over({}, data_dir=str(tmp_path))
    assert reader.config_file("config", default=None) is None
    assert problems(report) == []


def test_config_file_empty_path_is_problem(tmp_path):
    reader, report = reader_over({"config": ""}, data_dir=str(tmp_path))
    assert reader.config_file("config").failed
    assert problems(report) == [("block.config", "must not be empty; give a value")]
