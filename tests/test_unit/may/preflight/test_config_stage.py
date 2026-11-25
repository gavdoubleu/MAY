"""`ConfigStage` owns preflight and from_config; a stage supplies parse and build."""
import logging
import subprocess
import sys

import pytest

from may.preflight.config_stage import ConfigError, ConfigStage, PreflightError
from may.preflight.context import ConfigContext
from may.preflight.report import PreflightReport


class ToyDomainError(Exception):
    pass


class ToyConfigError(ConfigError, ToyDomainError):
    pass


class Plain(ConfigStage):
    """No dependencies: reads one required whole number."""

    display_name = "Plain"
    config_error = ToyConfigError
    config_key = "plain"

    @classmethod
    def parse(cls, reader, context):
        return {"size": reader.whole_number("size")}

    @classmethod
    def build(cls, parsed, **live):
        return ("built", parsed, live)


class ReadsContext(ConfigStage):
    """Reads Plain's result from the context."""

    display_name = "Reads context"
    config_error = ToyConfigError
    config_key = None

    @classmethod
    def parse(cls, reader, context):
        return context[Plain]

    @classmethod
    def build(cls, parsed, **live):
        return parsed


class WarnsOnly(ConfigStage):
    display_name = "Warns only"
    config_error = ToyConfigError
    config_key = "warns"

    @classmethod
    def parse(cls, reader, context):
        reader.warning("probably not what was meant", at="colour")
        reader.warning("also odd", at="shade")
        return "kept"

    @classmethod
    def build(cls, parsed, **live):
        return parsed


class ProblemButValue(ConfigStage):
    """Records problems yet returns a value; the base must null it."""

    display_name = "Problem"
    config_error = ToyConfigError
    config_key = "problem"

    @classmethod
    def parse(cls, reader, context):
        for count in range(reader.whole_number("count", default=1)):
            reader.problem(f"bad thing {count + 1}")
        return "half-checked"

    @classmethod
    def build(cls, parsed, **live):
        raise AssertionError("build must not run")


class ReturnsNone(ConfigStage):
    display_name = "Returns none"
    config_error = ToyConfigError
    config_key = "none"

    @classmethod
    def parse(cls, reader, context):
        return None

    @classmethod
    def build(cls, parsed, **live):
        return parsed


class Buggy(ConfigStage):
    display_name = "Buggy"
    config_error = ToyConfigError
    config_key = "buggy"

    @classmethod
    def parse(cls, reader, context):
        raise ZeroDivisionError("a bug")

    @classmethod
    def build(cls, parsed, **live):
        return parsed


EMPTY = ConfigContext()


def keys(findings):
    return [finding.key for finding in findings]


# --- preflight ---------------------------------------------------------------------------


def test_clean_config_gives_result_and_clean_report():
    result, report = Plain.preflight({"size": 3}, EMPTY)

    assert result == {"size": 3}
    assert report.problems == [] and report.warnings == []


def test_any_problem_nulls_the_result():
    result, report = Plain.preflight({}, EMPTY)

    assert result is None
    assert keys(report.problems) == ["plain.size"]


def test_problem_nulls_a_value_parse_returned():
    result, report = ProblemButValue.preflight({}, EMPTY)

    assert result is None
    assert keys(report.problems) == ["problem"]


def test_warnings_alone_keep_the_result():
    result, report = WarnsOnly.preflight({}, EMPTY)

    assert result == "kept"
    assert keys(report.warnings) == ["warns.colour", "warns.shade"]
    assert report.problems == []


def test_parse_returning_none_without_problems_is_a_type_error():
    with pytest.raises(TypeError, match="ReturnsNone"):
        ReturnsNone.preflight({}, EMPTY)


def test_root_key_path_defaults_to_config_key():
    _, report = Plain.preflight({}, EMPTY)

    assert keys(report.problems) == ["plain.size"]


def test_root_key_path_caller_override():
    _, report = Plain.preflight({}, EMPTY, key_path="timeline[3].config (foo.yaml)")

    assert keys(report.problems) == ["timeline[3].config (foo.yaml).size"]


def test_no_root_key_path_is_a_type_error():
    context = EMPTY.with_result(Plain, {"size": 1})

    with pytest.raises(TypeError):
        ReadsContext.preflight({}, context)


def test_step_stage_reads_its_dependency_from_the_context():
    context = EMPTY.with_result(Plain, {"size": 1})

    result, _ = ReadsContext.preflight({}, context, key_path="step")

    assert result == {"size": 1}


def test_blank_root_is_one_section_problem():
    result, report = Plain.preflight(None, EMPTY)

    assert result is None
    assert keys(report.problems) == ["plain"]
    assert report.problems[0].message.startswith("left blank")


@pytest.mark.parametrize("root", [["size"], 3, "size"])
def test_non_mapping_root_is_one_section_problem(root):
    result, report = Plain.preflight(root, EMPTY)

    assert result is None
    assert keys(report.problems) == ["plain"]
    assert "must be a mapping" in report.problems[0].message


def test_preflight_leaves_the_context_unchanged():
    context = EMPTY.with_result(Plain, {"size": 1})

    ReadsContext.preflight({}, context, key_path="step")

    assert list(context.stages()) == [Plain]


def test_preflight_requires_a_context():
    with pytest.raises(TypeError):
        Plain.preflight({"size": 3})


def test_unexpected_parse_exception_propagates():
    with pytest.raises(ZeroDivisionError, match="a bug"):
        Buggy.preflight({}, EMPTY)


# --- from_config -------------------------------------------------------------------------


def test_from_config_raises_the_declared_error():
    with pytest.raises(ToyConfigError) as raised:
        Plain.from_config({}, EMPTY)

    assert isinstance(raised.value, ConfigError)
    assert isinstance(raised.value, ToyDomainError)
    assert isinstance(raised.value.report, PreflightReport)
    assert keys(raised.value.report.problems) == ["plain.size"]


def test_from_config_message_is_uniform_header_and_body():
    with pytest.raises(ToyConfigError) as raised:
        Plain.from_config({}, EMPTY)

    report = raised.value.report
    assert str(raised.value) == f"Plain config cannot be used: 1 problem.\n\n{report.body()}"
    assert "plain.size: not set" in str(raised.value)


def test_from_config_header_plural():
    with pytest.raises(ToyConfigError, match=r"^Problem config cannot be used: 2 problems\."):
        ProblemButValue.from_config({"count": 2}, EMPTY)


def test_from_config_logs_each_warning_at_warning(caplog):
    with caplog.at_level(logging.WARNING):
        WarnsOnly.from_config({}, EMPTY)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "warns.colour: probably not what was meant" in warnings[0].getMessage()
    assert "warns.shade: also odd" in warnings[1].getMessage()


def test_from_config_builds_from_parsed_and_live_untouched():
    live_object = object()

    built = Plain.from_config({"size": 3}, EMPTY, geography=live_object)

    assert built == ("built", {"size": 3}, {"geography": live_object})
    assert built[2]["geography"] is live_object


def test_from_config_takes_key_path():
    with pytest.raises(ToyConfigError, match="elsewhere.size"):
        Plain.from_config({}, EMPTY, key_path="elsewhere")


def test_from_config_requires_a_context():
    with pytest.raises(TypeError):
        Plain.from_config({"size": 3})


# --- class definition --------------------------------------------------------------------


def test_missing_display_name_is_rejected():
    with pytest.raises(TypeError, match="display_name"):

        class NoName(ConfigStage):
            config_error = ToyConfigError

            @classmethod
            def parse(cls, reader, context):
                return 1

            @classmethod
            def build(cls, parsed, **live):
                return parsed


def test_empty_display_name_is_rejected():
    with pytest.raises(TypeError, match="display_name"):

        class EmptyName(ConfigStage):
            display_name = ""
            config_error = ToyConfigError

            @classmethod
            def parse(cls, reader, context):
                return 1

            @classmethod
            def build(cls, parsed, **live):
                return parsed


def test_missing_config_error_is_rejected():
    with pytest.raises(TypeError, match="config_error"):

        class NoError(ConfigStage):
            display_name = "No error"

            @classmethod
            def parse(cls, reader, context):
                return 1

            @classmethod
            def build(cls, parsed, **live):
                return parsed


@pytest.mark.parametrize("bad_error", [ValueError, "ToyConfigError", ToyConfigError("x", PreflightReport())])
def test_config_error_not_a_config_error_subclass_is_rejected(bad_error):
    with pytest.raises(TypeError, match="config_error"):

        class BadError(ConfigStage):
            display_name = "Bad error"
            config_error = bad_error

            @classmethod
            def parse(cls, reader, context):
                return 1

            @classmethod
            def build(cls, parsed, **live):
                return parsed


def test_abstract_intermediate_needs_no_declarations():
    class Intermediate(ConfigStage):
        @classmethod
        def build(cls, parsed, **live):
            return parsed

    class Concrete(Intermediate):
        display_name = "Concrete"
        config_error = ToyConfigError
        config_key = "concrete"

        @classmethod
        def parse(cls, reader, context):
            return reader.value()

    assert Concrete.from_config({"a": 1}, EMPTY) == {"a": 1}


def test_mixin_sits_beside_an_existing_base():
    class ExistingBase:
        def greet(self):
            return "hello"

    class Mixed(ExistingBase, ConfigStage):
        display_name = "Mixed"
        config_error = ToyConfigError
        config_key = "mixed"

        @classmethod
        def parse(cls, reader, context):
            return reader.value()

        @classmethod
        def build(cls, parsed, **live):
            return cls()

    assert Mixed.from_config({}, EMPTY).greet() == "hello"


# --- PreflightError ----------------------------------------------------------------------


def test_preflight_error_is_a_config_error_carrying_the_report():
    report = PreflightReport()
    report.problem("geography.levels", "not set; give a list")
    report.warning("venues", "odd")

    error = PreflightError(report)

    assert isinstance(error, ConfigError)
    assert error.report is report
    assert str(error) == report.format()


def test_preflight_error_old_import_path_still_works():
    from may.preflight.report import PreflightError as OldPathError

    assert OldPathError is PreflightError


@pytest.mark.parametrize(
    "module_order",
    [
        ["may.preflight.config_stage", "may.preflight.context", "may.preflight.report"],
        ["may.preflight.context", "may.preflight.config_stage"],
        ["may.preflight.report", "may.preflight.config_stage"],
        ["may.preflight.config_reader", "may.preflight.context"],
    ],
)
def test_no_import_cycle(module_order):
    imports = "; ".join(f"import {module}" for module in module_order)
    completed = subprocess.run(
        [sys.executable, "-c", f"{imports}; from may.preflight.report import PreflightError"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
