"""Config reader: how every config stage reads its config.

A reader is one place in the config (its key path) plus the stage parse's shared report.
Blank, absent and wrong-type values mean the same thing at every key in every stage:

- blank (`key:`, `null`, `~`) is always a problem;
- absent takes the default silently, or is a problem when no `default=` is given;
- wrong type is a problem; the only coercion is a whole number read as text.

A key with a problem reads as `FAILED`. A failed section or config file gives an inert
reader whose reads return `FAILED` and record nothing, so one mistake gives one finding.
"""

import yaml

from may.preflight.report import PreflightReport
from may.utils import path_resolver as pr
from may.utils.yaml_loader import invalid_yaml_message, load_yaml, unreadable_file_message

_SHOWN_LENGTH = 40
_QUOTE_HINT = "put it in quotes to use it as text"
_DEFAULT_NOUN = "a value"


class _Failed:
    """A read that had a problem: neither absent nor its default."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "FAILED"

    def __bool__(self):
        raise TypeError("a failed config read has no truth value; test `is FAILED`")


FAILED = _Failed()


class _Marker:
    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return self._name


_REQUIRED = _Marker("_REQUIRED")
_INERT = _Marker("_INERT")
_WHOLE_VALUE = _Marker("_WHOLE_VALUE")


def _is_whole_number(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _whole_numbers_as_text(spec):
    """A data-file spec with whole-number paths (alone or in a list) read as text."""
    if _is_whole_number(spec):
        return str(spec)
    if isinstance(spec, list):
        return [str(item) if _is_whole_number(item) else item for item in spec]
    return spec


def _shown(value) -> str:
    """A value as a message shows it: its YAML term, with the value for scalars."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return "a mapping"
    if isinstance(value, (list, tuple)):
        return "a list"
    text = repr(value)
    if len(text) > _SHOWN_LENGTH:
        text = text[:_SHOWN_LENGTH] + "…"
    if isinstance(value, str):
        return f"text {text}"
    if isinstance(value, int):
        return f"a whole number {text}"
    if isinstance(value, float):
        return f"a number {text}"
    return text


def _wrong_type(expected: str, value, hint: str = None) -> str:
    message = f"must be {expected}; got {_shown(value)}"
    return f"{message}; {hint}" if hint else message


class ConfigReader:
    """One place in the config. Reads record findings at their key path in the shared report."""

    def __init__(self, value, key_path: str, report: PreflightReport, data_dir=None):
        self._value = value
        self.key_path = key_path
        self._report = report
        self._data_dir = data_dir
        # A config file's own keys follow its path after `: `, not `.`.
        self._file_root = False
        self._found_path = None

    @property
    def found_path(self):
        """Where a `config_file` reader's file was found; None for any other reader."""
        return self._found_path

    @property
    def failed(self) -> bool:
        """True only for an inert reader (a failed or missing section or config file)."""
        return self._value is _INERT

    # --- key paths and findings --------------------------------------------------------

    def _child_path(self, key) -> str:
        key = str(key)
        if key.startswith("["):
            return f"{self.key_path}{key}"
        if not self.key_path:
            return key
        separator = ": " if self._file_root else "."
        return f"{self.key_path}{separator}{key}"

    def _item_path(self, parent_path: str, position: int) -> str:
        return f"{parent_path}[{position}]"

    def _reader_at(self, value, key_path: str) -> "ConfigReader":
        return ConfigReader(value, key_path, self._report, data_dir=self._data_dir)

    def _record(self, key_path: str, message: str) -> None:
        self._report.problem(key_path, message)

    def problem(self, message: str, at=None) -> None:
        """Record a domain problem here, or `at=` a child key or list position `[n]`."""
        self._record(self.key_path if at is None else self._child_path(at), message)

    def warning(self, message: str, at=None) -> None:
        self._report.warning(self.key_path if at is None else self._child_path(at), message)

    # --- the read shared by every primitive --------------------------------------------

    def _read(self, key, default, give, convert, convert_default=False):
        """Inert → FAILED; absent → default or problem; blank → problem; else `convert`."""
        if self.failed:
            return FAILED
        key_path = self._child_path(key)
        noun = give or _DEFAULT_NOUN
        if key not in self._value:
            if default is _REQUIRED:
                self._record(key_path, f"not set; give {noun}")
                return FAILED
            if default is None or not convert_default:
                return default
            return convert(default, key_path)
        raw = self._value[key]
        if raw is None:
            if default is _REQUIRED:
                self._record(key_path, f"left blank; give {noun}")
            else:
                self._record(
                    key_path, f"left blank; give {_DEFAULT_NOUN}, or remove the key to use the default"
                )
            return FAILED
        return convert(raw, key_path)

    def _wrong(self, key_path: str, expected: str, value, hint: str = None):
        self._record(key_path, _wrong_type(expected, value, hint))
        return FAILED

    def _to_text(self, raw, key_path):
        if isinstance(raw, str):
            return raw
        if _is_whole_number(raw):
            return str(raw)
        hint = _QUOTE_HINT if isinstance(raw, (bool, float)) else None
        return self._wrong(key_path, "text", raw, hint)

    def _to_name(self, raw, key_path, noun):
        text = self._to_text(raw, key_path)
        if text is FAILED:
            return FAILED
        if not text:
            self._record(key_path, f"must not be empty; give {noun}")
            return FAILED
        return text

    # --- navigation --------------------------------------------------------------------

    def section(self, key, default=_REQUIRED, give=None) -> "ConfigReader":
        """Reader over the mapping at `key`; inert when it is failed or required and absent;
        None when absent with `default=None`."""

        def convert(raw, key_path):
            if not isinstance(raw, dict):
                return self._wrong(key_path, "a mapping", raw)
            return self._reader_at(raw, key_path)

        result = self._read(key, default, give, convert, convert_default=True)
        if result is FAILED:
            return self._reader_at(_INERT, self._child_path(key))
        return result

    def items(self, key, default=_REQUIRED, give=None):
        """Readers for each mapping in the list at `key`; a non-mapping item is a problem, left out."""

        def convert(raw, key_path):
            if not isinstance(raw, list):
                return self._wrong(key_path, "a list", raw)
            readers = []
            for position, item in enumerate(raw, start=1):
                item_path = self._item_path(key_path, position)
                if isinstance(item, dict):
                    readers.append(self._reader_at(item, item_path))
                else:
                    self._wrong(item_path, "a mapping", item)
            return readers

        return self._read(key, default, give, convert, convert_default=True)

    def entries(self, key, default=_REQUIRED, give=None):
        """Name → reader for a mapping of mappings; whole-number names read as text.

        A name not text, or the same as an earlier one once read as text, is a problem, left out."""

        def convert(raw, key_path):
            if not isinstance(raw, dict):
                return self._wrong(key_path, "a mapping", raw)
            readers = {}
            entry_parent = self._reader_at(raw, key_path)
            for name, entry in raw.items():
                entry_path = entry_parent._child_path(name)
                if not isinstance(name, str) and not _is_whole_number(name):
                    hint = _QUOTE_HINT if isinstance(name, (bool, float)) else None
                    self._record(entry_path, "entry name " + _wrong_type("text", name, hint))
                elif str(name) in readers:
                    self._record(entry_path, "named twice; give each entry a different name")
                elif isinstance(entry, dict):
                    readers[str(name)] = self._reader_at(entry, entry_path)
                else:
                    self._wrong(entry_path, "a mapping", entry)
            return readers

        return self._read(key, default, give, convert, convert_default=True)

    def keys(self):
        """Names written at this place, as written."""
        if self.failed:
            return FAILED
        return list(self._value)

    def config_file(self, key, data_dir=_WHOLE_VALUE, default=_REQUIRED, give=None) -> "ConfigReader":
        """Reader over the YAML file `key` names, found once and loaded once.

        Found absolute as written, else in the working dir, else under the data dir
        (`data_dir=` or this reader's). Every failure is one problem at `key` and gives
        an inert reader. A failed data dir gives an inert reader with nothing recorded.
        Absent with `default=None` gives None."""
        key_path = self._child_path(key)
        written_path = self.name(key, default=default, give=give)
        if data_dir is _WHOLE_VALUE:
            data_dir = self._data_dir
        if written_path is None:
            return None
        if written_path is FAILED or data_dir is FAILED:
            return self._reader_at(_INERT, key_path)
        config = self._load_config_file(written_path, data_dir, key_path)
        if config is FAILED:
            return self._reader_at(_INERT, key_path)
        found_path, mapping = config
        file_reader = ConfigReader(mapping, f"{key_path} ({found_path})", self._report, data_dir)
        file_reader._file_root = True
        file_reader._found_path = found_path
        return file_reader

    def _load_config_file(self, written_path, data_dir, key_path):
        try:
            found_path = pr.find_config_file(written_path, data_dir)
        except pr.ConfigFileNotFoundError as error:
            self._record(key_path, str(error))
            return FAILED
        try:
            config = load_yaml(found_path)
        except yaml.YAMLError as error:
            self._record(key_path, invalid_yaml_message(found_path, error))
            return FAILED
        except (OSError, UnicodeDecodeError) as error:
            self._record(key_path, unreadable_file_message(found_path, error))
            return FAILED
        if config is None:
            self._record(key_path, f"{found_path} is empty")
            return FAILED
        if not isinstance(config, dict):
            self._record(key_path, f"{found_path} must hold a mapping; got {_shown(config)}")
            return FAILED
        return found_path, config

    def root_reader(self, key_path: str, value, give=None) -> "ConfigReader":
        """Reader at absolute `key_path` over `value`, same report and data dir.

        Blank or non-mapping `value` is one problem at `key_path`; the reader is then inert."""
        if value is None:
            self._record(key_path, f"left blank; give {give or _DEFAULT_NOUN}")
            return self._reader_at(_INERT, key_path)
        if not isinstance(value, dict):
            self._record(key_path, _wrong_type("a mapping", value))
            return self._reader_at(_INERT, key_path)
        return self._reader_at(value, key_path)

    # --- leaves ------------------------------------------------------------------------

    def text(self, key, default=_REQUIRED, give=None):
        """Text; a whole number is read as text."""
        return self._read(key, default, give, self._to_text)

    def name(self, key, default=_REQUIRED, give=None):
        """Non-empty text; a whole number is read as text."""
        return self._read(
            key, default, give, lambda raw, key_path: self._to_name(raw, key_path, give or _DEFAULT_NOUN)
        )

    def names(self, key, default=_REQUIRED, give=None):
        """List of names; a bad item is a problem at `key[n]`, left out. Duplicates are the stage's."""

        def convert(raw, key_path):
            if not isinstance(raw, list):
                return self._wrong(key_path, "a list", raw)
            good_names = []
            for position, item in enumerate(raw, start=1):
                item_path = self._item_path(key_path, position)
                if item is None:
                    self._record(item_path, "left blank; give a name")
                    continue
                name = self._to_name(item, item_path, "a name")
                if name is not FAILED:
                    good_names.append(name)
            return good_names

        return self._read(key, default, give, convert)

    def true_or_false(self, key, default=_REQUIRED, give=None):
        def convert(raw, key_path):
            return raw if isinstance(raw, bool) else self._wrong(key_path, "true or false", raw)

        return self._read(key, default, give, convert)

    def whole_number(self, key, default=_REQUIRED, give=None):
        def convert(raw, key_path):
            return raw if _is_whole_number(raw) else self._wrong(key_path, "a whole number", raw)

        return self._read(key, default, give, convert)

    def number(self, key, default=_REQUIRED, give=None):
        """A number; whole numbers count, true/false does not."""

        def convert(raw, key_path):
            if isinstance(raw, float) or _is_whole_number(raw):
                return raw
            return self._wrong(key_path, "a number", raw)

        return self._read(key, default, give, convert)

    def one_of(self, key, allowed, default=_REQUIRED, give=None):
        """One of a closed vocabulary; a miss lists every allowed value, sorted."""

        def convert(raw, key_path):
            value = str(raw) if _is_whole_number(raw) else raw
            if isinstance(value, str) and value in allowed:
                return value
            choices = ", ".join(sorted(allowed))
            self._record(key_path, f"must be one of {choices}; got {_shown(raw)}")
            return FAILED

        return self._read(key, default, give, convert)

    def path(self, key, default=_REQUIRED, give=None):
        """Text with `${...}` roots resolved."""

        def convert(raw, key_path):
            text = self._to_text(raw, key_path)
            return FAILED if text is FAILED else pr.resolve(text)

        return self._read(key, default, give, convert)

    def data_file(self, key, data_dir, default=_REQUIRED, give=None):
        """Paths of the data file(s) `key` names: one path or a list, relative ones under
        `data_dir`, each must exist; whole numbers read as text. A failed `data_dir` checks
        the shape only. A default is found and checked as if written."""

        def convert(raw, key_path):
            raw = _whole_numbers_as_text(raw)
            try:
                pr.as_path_list(raw, key_path)
            except pr.DataFileSpecError:
                self._record(
                    key_path, f"must be a path or a non-empty list of paths; got {_shown(raw)}"
                )
                return FAILED
            if data_dir is FAILED:
                return FAILED
            try:
                return pr.data_file_paths(raw, data_dir, key_path)
            except pr.DataFileNotFoundError as error:
                self._record(key_path, str(error))
                return FAILED

        return self._read(key, default, give, convert, convert_default=True)

    def value(self, key=_WHOLE_VALUE, default=None):
        """Raw value at `key` (or this whole place), recording nothing; `FAILED` when inert."""
        if self.failed:
            return FAILED
        if key is _WHOLE_VALUE:
            return self._value
        return self._value.get(key, default)
