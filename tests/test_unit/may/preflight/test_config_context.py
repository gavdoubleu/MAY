"""`ConfigContext`: immutable, build-ordered map from config stage class to parsed result."""
import pytest

from may.preflight.config_stage import ConfigError, ConfigStage
from may.preflight.context import ConfigContext, StageNotParsed


class ToyConfigError(ConfigError):
    pass


class _ToyStage(ConfigStage):
    display_name = "Toy"
    config_error = ToyConfigError

    @classmethod
    def parse(cls, reader, context):
        return reader.value()

    @classmethod
    def build(cls, parsed, **live):
        return parsed


class First(_ToyStage):
    display_name = "First"
    config_key = "first"


class Second(_ToyStage):
    display_name = "Second"
    config_key = "second"


class FirstSubclass(First):
    display_name = "First subclass"


class NotAStage:
    pass


def test_lookup_returns_the_stored_result():
    context = ConfigContext(results={First: {"levels": ["SGU"]}})

    assert context[First] == {"levels": ["SGU"]}


def test_failed_entry_is_none_and_present():
    context = ConfigContext(results={First: None})

    assert context[First] is None
    assert First in context


def test_absent_lookup_raises_stage_not_parsed_naming_the_stage():
    context = ConfigContext()

    with pytest.raises(StageNotParsed, match="First") as raised:
        context[First]

    assert isinstance(raised.value, KeyError)
    assert "not been parsed" in str(raised.value)
    assert First not in context


def test_with_result_returns_new_context_and_leaves_original():
    original = ConfigContext(results={First: 1})

    extended = original.with_result(Second, 2)

    assert extended[Second] == 2 and extended[First] == 1
    assert Second not in original
    assert list(original.stages()) == [First]


def test_with_result_carries_world_config():
    original = ConfigContext(world_config={"households": {}})

    assert original.with_result(First, 1).world_config == {"households": {}}


def test_duplicate_stage_raises():
    context = ConfigContext(results={First: 1})

    with pytest.raises(ValueError, match="First"):
        context.with_result(First, 2)


@pytest.mark.parametrize("bad_key", ["First", NotAStage, First(), ConfigStage])
def test_non_stage_key_is_a_type_error_via_with_result(bad_key):
    with pytest.raises(TypeError):
        ConfigContext().with_result(bad_key, 1)


@pytest.mark.parametrize("bad_key", ["First", NotAStage, First()])
def test_non_stage_key_is_a_type_error_via_constructor(bad_key):
    with pytest.raises(TypeError):
        ConfigContext(results={bad_key: 1})


def test_entries_cannot_be_set_or_deleted():
    context = ConfigContext(results={First: 1})

    with pytest.raises(TypeError):
        context[Second] = 2
    with pytest.raises(TypeError):
        del context[First]


def test_attributes_cannot_be_set():
    context = ConfigContext()

    with pytest.raises(AttributeError):
        context.world_config = {"geography": {}}
    with pytest.raises(AttributeError):
        context.anything = 1


def test_constructor_copies_results():
    results = {First: 1}
    context = ConfigContext(results=results)

    results[Second] = 2

    assert Second not in context


def test_world_config_defaults_to_empty():
    assert dict(ConfigContext().world_config) == {}


def test_world_config_is_read_only():
    context = ConfigContext(world_config={"geography": {"levels": []}})

    with pytest.raises(TypeError):
        context.world_config["geography"] = None
    assert context.world_config["geography"] == {"levels": []}


def test_stages_keep_insertion_order():
    context = ConfigContext(results={Second: 2}).with_result(First, 1)

    assert list(context.stages()) == [Second, First]


def test_lookup_is_by_exact_class():
    context = ConfigContext(results={First: 1})

    assert FirstSubclass not in context
    with pytest.raises(StageNotParsed):
        context[FirstSubclass]
