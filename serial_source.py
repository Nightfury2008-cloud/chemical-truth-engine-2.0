"""Optional ESP32 USB serial input for Chemical Truth Engine 2.0.

SAFETY AND HONESTY
------------------
* This module is READ-ONLY. It never writes to, configures or commands the
  device. It does not operate pumps, relays, dosing equipment or valves.
* It never connects automatically. The user must explicitly list ports,
  choose one, and press Connect.
* pySerial is OPTIONAL. If it is not installed, every entry point returns a
  clear, friendly error instead of crashing, and the simulation / CSV modes
  keep working.
* No hardware was available during development, so the read path against a
  real ESP32 has NOT been exercised. The parsing and connection-failure logic
  are unit tested with a fake serial object; treat live hardware as untested.

MESSAGE FORMAT
--------------
One JSON object per line (newline-delimited), for example::

    {"ph": 8.02, "conductivity": 648.5, "temperature": 25.4,
     "timestamp": "2026-01-01T12:00:05", "claimed_state": "Steady operation"}

``ph``, ``conductivity`` and ``temperature`` are required and must be finite
numbers (pH dimensionless, conductivity in uS/cm, temperature in degrees C).
``timestamp`` and ``claimed_state`` are optional. Every line is validated;
malformed lines are rejected and counted, never guessed at.

This module is free of any Streamlit import so it can be unit tested directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any, Callable, Mapping

from detection_engine import (
    LABELS,
    MEASUREMENT_KEYS,
    PLAUSIBLE_RANGES,
    UNITS,
)
from simulation import QUALITY_OK

__all__ = [
    "DEFAULT_BAUDRATE",
    "DEFAULT_STALENESS_SECONDS",
    "PortInfo",
    "SerialMessage",
    "SerialReader",
    "age_seconds",
    "is_stale",
    "list_ports",
    "parse_message",
    "pyserial_available",
]

DEFAULT_BAUDRATE = 115200
DEFAULT_STALENESS_SECONDS = 10.0


# --------------------------------------------------------------------------- #
# pySerial availability (optional dependency)
# --------------------------------------------------------------------------- #

def _import_serial() -> tuple[Any | None, str | None]:
    """Import pySerial lazily. Returns (module, error_message)."""
    try:
        import serial  # type: ignore
        import serial.tools.list_ports as list_ports_mod  # type: ignore
    except ImportError:
        return None, (
            "pySerial is not installed, so ESP32 USB serial input is "
            "unavailable. Simulation and CSV replay still work. To enable "
            "serial input, run: python -m pip install pyserial"
        )
    # Expose list_ports on the module for convenience.
    serial.list_ports = list_ports_mod
    return serial, None


def pyserial_available() -> bool:
    """True when the optional pySerial dependency can be imported."""
    module, _ = _import_serial()
    return module is not None


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PortInfo:
    """A detected serial port."""

    device: str
    description: str


@dataclass(frozen=True)
class SerialMessage:
    """A validated reading parsed from one serial line."""

    observed: dict[str, float]
    quality: dict[str, str]
    timestamp: datetime | None
    claimed_state: str | None


# --------------------------------------------------------------------------- #
# Port listing
# --------------------------------------------------------------------------- #

def list_ports() -> tuple[list[PortInfo], str | None]:
    """List detected serial ports.

    Returns ``(ports, error)``. Never raises; when pySerial is missing or
    listing fails, ``ports`` is empty and ``error`` explains why.
    """
    serial, err = _import_serial()
    if err:
        return [], err
    try:
        found = list(serial.list_ports.comports())
    except Exception as exc:  # pragma: no cover - driver-specific
        return [], f"Could not list serial ports: {exc}"
    ports = [
        PortInfo(device=p.device, description=getattr(p, "description", ""))
        for p in found
    ]
    return ports, None


# --------------------------------------------------------------------------- #
# Message parsing / validation
# --------------------------------------------------------------------------- #

def _parse_timestamp(raw: Any) -> datetime | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).strip())
    except ValueError:
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def parse_message(
    line: str,
    *,
    default_claimed_state: str | None = None,
) -> tuple[SerialMessage | None, list[str]]:
    """Parse and validate one newline-delimited JSON message.

    Returns ``(message, errors)``. On any validation failure ``message`` is
    None and ``errors`` describes every problem found. Nothing is guessed.
    """
    errors: list[str] = []
    if line is None or not str(line).strip():
        return None, ["Received an empty line (no data)."]

    text = str(line).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"Line is not valid JSON: {exc}."]
    if not isinstance(payload, Mapping):
        return None, [
            f"Expected a JSON object of measurements, got "
            f"{type(payload).__name__}."
        ]

    observed: dict[str, float] = {}
    quality: dict[str, str] = {}
    for key in MEASUREMENT_KEYS:
        if key not in payload:
            errors.append(
                f"Missing required field {key!r} "
                f"({LABELS[key]} in {UNITS[key]})."
            )
            continue
        raw = payload[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errors.append(
                f"Field {key!r} must be a number (received {raw!r})."
            )
            continue
        value = float(raw)
        if not isfinite(value):
            errors.append(f"Field {key!r} must be finite (received {raw!r}).")
            continue
        low, high = PLAUSIBLE_RANGES[key]
        if not low <= value <= high:
            errors.append(
                f"Field {key!r} value {value} is outside the plausible range "
                f"{low} to {high} {UNITS[key]}."
            )
            continue
        observed[key] = value
        quality[key] = QUALITY_OK

    if errors:
        return None, errors

    timestamp = _parse_timestamp(payload.get("timestamp"))
    if payload.get("timestamp") is not None and timestamp is None:
        # A present-but-unparseable timestamp is a warning, not fatal: the
        # reading itself is still valid, just unanchored in time.
        pass

    claim_raw = payload.get("claimed_state")
    claimed_state = (
        str(claim_raw).strip()
        if claim_raw is not None and str(claim_raw).strip()
        else default_claimed_state
    )

    return (
        SerialMessage(
            observed=observed,
            quality=quality,
            timestamp=timestamp,
            claimed_state=claimed_state,
        ),
        [],
    )


# --------------------------------------------------------------------------- #
# Freshness helpers
# --------------------------------------------------------------------------- #

def age_seconds(
    last_valid_at: datetime | None,
    now: datetime,
) -> float | None:
    """Seconds since the last valid sample, or None if none yet."""
    if last_valid_at is None:
        return None
    return max(0.0, (now - last_valid_at).total_seconds())


def is_stale(
    last_valid_at: datetime | None,
    now: datetime,
    max_age_seconds: float = DEFAULT_STALENESS_SECONDS,
) -> bool:
    """True when there is no data yet or the newest data is too old."""
    age = age_seconds(last_valid_at, now)
    if age is None:
        return True
    return age > max_age_seconds


# --------------------------------------------------------------------------- #
# Read-only serial reader
# --------------------------------------------------------------------------- #

class SerialReader:
    """Read-only wrapper around a single serial port.

    The caller explicitly connects to a chosen port. The reader only ever
    calls ``readline``; it never writes to the device.
    """

    def __init__(
        self,
        *,
        now_fn: Callable[[], datetime] | None = None,
        serial_module: Any | None = None,
    ) -> None:
        self._now = now_fn or datetime.now
        self._serial_module = serial_module
        self._port: Any | None = None
        self._device: str | None = None
        self.malformed_count = 0
        self.last_valid_at: datetime | None = None

    # ------------------------------------------------------------------ #
    @property
    def connected(self) -> bool:
        return self._port is not None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def last_valid_age_seconds(self) -> float | None:
        return age_seconds(self.last_valid_at, self._now())

    def connect(
        self,
        device: str,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = 2.0,
    ) -> str | None:
        """Open the chosen port. Returns an error message, or None on success.

        Never called automatically; the user must request it.
        """
        if self.connected:
            return "Already connected. Disconnect first."
        if not device or not str(device).strip():
            return "No serial port was selected."

        serial = self._serial_module
        if serial is None:
            serial, err = _import_serial()
            if err:
                return err
        try:
            self._port = serial.Serial(
                port=str(device),
                baudrate=int(baudrate),
                timeout=float(timeout),
            )
        except Exception as exc:
            return f"Could not open {device}: {exc}"
        self._device = str(device)
        return None

    def disconnect(self) -> None:
        """Close the port if open. Safe to call repeatedly."""
        if self._port is not None:
            try:
                self._port.close()
            except Exception:  # pragma: no cover - driver-specific
                pass
        self._port = None
        self._device = None

    def read_line(self) -> tuple[str | None, str | None]:
        """Read one line. Returns ``(line, error)``; never raises."""
        if not self.connected:
            return None, "Not connected to a serial port."
        try:
            raw = self._port.readline()
        except Exception as exc:
            return None, f"Serial read failed: {exc}"
        if raw is None or raw == b"":
            return None, "No data received before the timeout."
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw).strip()
        if not line:
            return None, "No data received before the timeout."
        return line, None

    def read_message(
        self,
        *,
        default_claimed_state: str | None = None,
    ) -> tuple[SerialMessage | None, list[str]]:
        """Read and validate one message.

        Returns ``(message, errors)``. On success, records the validity
        timestamp. Malformed lines increment ``malformed_count``.
        """
        line, read_err = self.read_line()
        if read_err:
            return None, [read_err]
        message, errors = parse_message(
            line, default_claimed_state=default_claimed_state
        )
        if message is not None:
            self.last_valid_at = self._now()
        else:
            self.malformed_count += 1
        return message, errors
