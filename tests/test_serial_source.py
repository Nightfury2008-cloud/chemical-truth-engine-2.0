"""Unit tests for serial_source - the optional ESP32 serial integration.

These tests run WITHOUT hardware and WITHOUT pySerial: the connection logic
is exercised through an injected fake serial module, and message parsing is
pure. This mirrors the requirement that the app must be testable without a
physical ESP32 and must degrade gracefully when pySerial is absent.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from detection_engine import VerificationStatus, verify
from serial_source import (
    DEFAULT_BAUDRATE,
    SerialReader,
    age_seconds,
    is_stale,
    list_ports,
    parse_message,
    pyserial_available,
)
from simulation import DEFAULT_EXPECTED, DEFAULT_TOLERANCES

GOOD_LINE = (
    '{"ph": 8.02, "conductivity": 648.5, "temperature": 25.4, '
    '"timestamp": "2026-01-01T12:00:05", "claimed_state": "Steady operation"}'
)


# --------------------------------------------------------------------------- #
# Message parsing / validation (pure, no hardware)
# --------------------------------------------------------------------------- #

def test_good_message_parses():
    message, errors = parse_message(GOOD_LINE)
    assert errors == []
    assert message is not None
    assert message.observed["ph"] == pytest.approx(8.02)
    assert message.quality["ph"] == "ok"
    assert message.timestamp == datetime(2026, 1, 1, 12, 0, 5)
    assert message.claimed_state == "Steady operation"


def test_message_without_optional_fields_parses():
    line = '{"ph": 8.0, "conductivity": 650.0, "temperature": 25.5}'
    message, errors = parse_message(line, default_claimed_state="Steady")
    assert errors == []
    assert message.timestamp is None
    assert message.claimed_state == "Steady"


def test_empty_line_is_rejected():
    for line in ("", "   ", None):
        message, errors = parse_message(line)
        assert message is None
        assert errors


def test_non_json_line_is_rejected():
    message, errors = parse_message("hello world")
    assert message is None
    assert any("JSON" in e for e in errors)


def test_json_array_is_rejected():
    message, errors = parse_message("[1, 2, 3]")
    assert message is None
    assert errors


def test_missing_field_is_rejected():
    message, errors = parse_message('{"ph": 8.0, "temperature": 25.5}')
    assert message is None
    assert any("conductivity" in e for e in errors)


def test_nonnumeric_field_is_rejected():
    line = '{"ph": "acid", "conductivity": 650.0, "temperature": 25.5}'
    message, errors = parse_message(line)
    assert message is None
    assert any("must be a number" in e for e in errors)


def test_nan_inf_and_bool_fields_are_rejected():
    for bad in ("null", "true", '"8.0"'):
        line = f'{{"ph": {bad}, "conductivity": 650.0, "temperature": 25.5}}'
        message, errors = parse_message(line)
        assert message is None, bad


def test_out_of_range_field_is_rejected():
    line = '{"ph": 99.0, "conductivity": 650.0, "temperature": 25.5}'
    message, errors = parse_message(line)
    assert message is None
    assert any("plausible range" in e for e in errors)


def test_parsed_message_verifies_normal_under_defaults():
    message, _ = parse_message(GOOD_LINE)
    r = verify(
        message.observed,
        DEFAULT_EXPECTED,
        DEFAULT_TOLERANCES,
        message.claimed_state,
        data_quality=message.quality,
    )
    assert r.status is VerificationStatus.NORMAL


# --------------------------------------------------------------------------- #
# Freshness helpers
# --------------------------------------------------------------------------- #

def test_age_seconds_none_when_no_data():
    now = datetime(2026, 1, 1, 12, 0, 10)
    assert age_seconds(None, now) is None
    assert is_stale(None, now, 10.0) is True


def test_age_seconds_grows_and_is_never_negative():
    now = datetime(2026, 1, 1, 12, 0, 10)
    last = now - timedelta(seconds=4)
    assert age_seconds(last, now) == pytest.approx(4.0)
    # A clock skew into the future clamps to 0 rather than going negative.
    assert age_seconds(now + timedelta(seconds=5), now) == 0.0


def test_is_stale_boundary_is_inclusive():
    now = datetime(2026, 1, 1, 12, 0, 10)
    assert is_stale(now - timedelta(seconds=10), now, 10.0) is False
    assert is_stale(now - timedelta(seconds=10.5), now, 10.0) is True


# --------------------------------------------------------------------------- #
# Connection logic via an injected fake serial module (no hardware)
# --------------------------------------------------------------------------- #

class FakeSerial:
    """Duck-typed stand-in for serial.Serial - read-only in behaviour."""

    def __init__(self, port=None, baudrate=9600, timeout=1.0, lines=None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.lines = list(lines or [])
        self.closed = False
        self.written = []  # the reader must never write

    def readline(self):
        return self.lines.pop(0) if self.lines else b""

    def write(self, data):  # pragma: no cover - must never be called
        self.written.append(data)
        return len(data)

    def close(self):
        self.closed = True


def _fake_module(lines=None, fail_open=False):
    def _factory(**kwargs):
        if fail_open:
            raise OSError("port busy")
        instance = FakeSerial(lines=lines, **kwargs)
        _factory.last = instance
        return instance

    _factory.last = None
    return SimpleNamespace(Serial=_factory)


def test_connect_and_read_one_sample():
    module = _fake_module(lines=[GOOD_LINE.encode() + b"\n"])
    reader = SerialReader(serial_module=module)
    assert reader.connected is False
    error = reader.connect("COM3", baudrate=DEFAULT_BAUDRATE)
    assert error is None
    assert reader.connected is True
    assert reader.device == "COM3"

    message, errors = reader.read_message(default_claimed_state="Steady")
    assert errors == []
    assert message is not None
    assert message.observed["conductivity"] == pytest.approx(648.5)
    assert reader.last_valid_at is not None
    assert reader.malformed_count == 0


def test_reader_never_writes_to_the_device():
    module = _fake_module(lines=[GOOD_LINE.encode() + b"\n"])
    reader = SerialReader(serial_module=module)
    reader.connect("COM3")
    reader.read_message()
    reader.disconnect()
    assert module.Serial.last.written == []


def test_connect_failure_returns_error_not_exception():
    module = _fake_module(fail_open=True)
    reader = SerialReader(serial_module=module)
    error = reader.connect("COM3")
    assert error is not None
    assert reader.connected is False


def test_connect_with_no_port_selected():
    reader = SerialReader(serial_module=_fake_module())
    assert reader.connect("") is not None
    assert reader.connect(None) is not None


def test_double_connect_is_refused():
    reader = SerialReader(serial_module=_fake_module())
    reader.connect("COM3")
    assert reader.connect("COM4") is not None
    assert reader.device == "COM3"


def test_read_without_connection_is_an_error():
    reader = SerialReader(serial_module=_fake_module())
    message, errors = reader.read_message()
    assert message is None
    assert any("Not connected" in e for e in errors)


def test_timeout_returns_friendly_error():
    reader = SerialReader(serial_module=_fake_module(lines=[]))
    reader.connect("COM3")
    message, errors = reader.read_message()
    assert message is None
    assert any("timeout" in e.lower() for e in errors)


def test_malformed_line_is_counted_not_guessed():
    reader = SerialReader(serial_module=_fake_module(lines=[b"garbage\n"]))
    reader.connect("COM3")
    message, errors = reader.read_message()
    assert message is None
    assert reader.malformed_count == 1


def test_partial_message_is_counted_as_malformed():
    line = b'{"ph": 8.0}\n'
    reader = SerialReader(serial_module=_fake_module(lines=[line]))
    reader.connect("COM3")
    message, errors = reader.read_message()
    assert message is None
    assert reader.malformed_count == 1
    assert errors


def test_disconnect_is_idempotent():
    module = _fake_module()
    reader = SerialReader(serial_module=module)
    reader.connect("COM3")
    port = module.Serial.last
    reader.disconnect()
    reader.disconnect()  # second call must not raise
    assert reader.connected is False
    assert port.closed is True


def test_stale_detection_via_injected_clock():
    clock = {"now": datetime(2026, 1, 1, 12, 0, 0)}
    reader = SerialReader(
        serial_module=_fake_module(lines=[GOOD_LINE.encode() + b"\n"]),
        now_fn=lambda: clock["now"],
    )
    reader.connect("COM3")
    reader.read_message()
    assert reader.last_valid_age_seconds == 0.0
    clock["now"] += timedelta(seconds=3)
    assert reader.last_valid_age_seconds == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Graceful degradation when pySerial / hardware is absent
# --------------------------------------------------------------------------- #

def test_pyserial_availability_is_boolean():
    assert isinstance(pyserial_available(), bool)


def test_list_ports_returns_tuple_and_never_raises():
    ports, error = list_ports()
    assert isinstance(ports, list)
    # In this environment pySerial is not installed, so an error is expected;
    # on machines where it is installed, a list (possibly empty) is returned.
    if error is not None:
        assert "pySerial" in error
    for p in ports:
        assert p.device


def test_default_baudrate_is_sane():
    assert DEFAULT_BAUDRATE == 115200
