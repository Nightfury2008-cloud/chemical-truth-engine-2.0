"""CSV replay: validate and load recorded measurements for replay.

This module is free of any Streamlit import so its validation logic can be
unit tested directly. It reads a CSV of recorded measurements and produces a
list of timestamped records that the dashboard replays one sample at a time.

CSV FORMAT
----------
Required columns (case-insensitive; extra columns are ignored):

* ``timestamp``      - an ISO-8601 date/time string, e.g. ``2026-01-01T12:00:05``
* ``ph``             - pH, dimensionless
* ``conductivity``   - conductivity in uS/cm
* ``temperature``   - temperature in degrees C

Optional column:

* ``claimed_state``  - the controller-claimed state for that row; falls back
  to ``default_claimed_state`` when absent or blank.

HONESTY NOTE
------------
Imported rows are RECORDED data. They are replayed and compared against the
configured expected response, but they are NOT verified, calibrated or
experimental measurements. This module validates structure and data types
only; it cannot vouch for how the numbers were produced.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Mapping

from detection_engine import (
    LABELS,
    MEASUREMENT_KEYS,
    PLAUSIBLE_RANGES,
    UNITS,
)
from simulation import (
    QUALITY_INVALID,
    QUALITY_MISSING,
    QUALITY_OK,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "CSVRecord",
    "CsvLoadResult",
    "parse_csv",
]

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ph",
    "conductivity",
    "temperature",
)

#: Optional alias -> canonical measurement key (header spelling flexibility).
_HEADER_ALIASES: dict[str, str] = {
    "ph": "ph",
    "conductivity": "conductivity",
    "conductivity_uscm": "conductivity",
    "temperature": "temperature",
    "temp": "temperature",
    "timestamp": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
    "claimed_state": "claimed_state",
    "claim": "claimed_state",
}


@dataclass(frozen=True)
class CSVRecord:
    """One validated row from an uploaded CSV."""

    index: int                    # 0-based position among parsed records
    row_number: int               # 1-based data-row number (excluding header)
    timestamp: datetime | None
    observed: dict[str, float | None]
    quality: dict[str, str]
    claimed_state: str


@dataclass
class CsvLoadResult:
    """Outcome of parsing an uploaded CSV."""

    records: list[CSVRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_rows: int = 0

    @property
    def ok(self) -> bool:
        """True when at least one usable record was loaded."""
        return bool(self.records)


def _parse_timestamp(raw: Any) -> tuple[datetime | None, str | None]:
    """Parse an ISO-8601 timestamp. Returns (value, error)."""
    if raw is None or not str(raw).strip():
        return None, "timestamp is empty"
    text = str(raw).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None, f"timestamp {text!r} is not a valid ISO-8601 date/time"
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)  # normalise to naive local-style
    return parsed, None


def _parse_value(
    key: str, raw: Any
) -> tuple[float | None, str, str | None]:
    """Parse one measurement cell.

    Returns (value, quality, error). ``value`` is None whenever the cell is
    not a usable number; the quality records why.
    """
    label = LABELS[key]
    unit = UNITS[key]
    if raw is None or not str(raw).strip():
        return None, QUALITY_MISSING, f"{label} is empty"
    text = str(raw).strip()
    try:
        value = float(text)
    except ValueError:
        return None, QUALITY_INVALID, (
            f"{label} value {text!r} is not a number"
        )
    if not isfinite(value):
        return None, QUALITY_INVALID, (
            f"{label} value {text!r} is not a finite number"
        )
    low, high = PLAUSIBLE_RANGES[key]
    if not low <= value <= high:
        return None, QUALITY_INVALID, (
            f"{label} value {value} is outside the plausible range "
            f"{low} to {high} {unit}"
        )
    return value, QUALITY_OK, None


def parse_csv(
    text: str,
    *,
    default_claimed_state: str = "Steady operation",
) -> CsvLoadResult:
    """Parse and validate a CSV of recorded measurements.

    Rows with problems are kept with ``None`` readings and a non-``ok``
    quality flag so the engine reports INSUFFICIENT DATA for them; invalid
    data is never silently replaced with normal values. Problems are also
    collected in :attr:`CsvLoadResult.errors` for the user to read.
    """
    result = CsvLoadResult()

    if text is None or not text.strip():
        result.errors.append("The file is empty.")
        return result

    reader = csv.reader(io.StringIO(text))
    try:
        header_row = next(reader)
    except StopIteration:
        result.errors.append("The file is empty.")
        return result

    # Map header cells to canonical names.
    header_map: dict[int, str] = {}
    seen: set[str] = set()
    for idx, cell in enumerate(header_row):
        name = cell.strip().lower().replace(" ", "_")
        canonical = _HEADER_ALIASES.get(name)
        if canonical is None:
            continue  # ignore unknown/extra columns
        if canonical in seen:
            result.warnings.append(
                f"Duplicate column {cell.strip()!r}; using the first one."
            )
            continue
        seen.add(canonical)
        header_map[idx] = canonical

    missing = [c for c in REQUIRED_COLUMNS if c not in seen]
    if missing:
        result.errors.append(
            "Missing required column(s): " + ", ".join(missing) + "."
        )
        result.errors.append(
            "Expected columns: timestamp, ph, conductivity, temperature "
            "(optional: claimed_state). Conductivity must be in uS/cm and "
            "temperature in degrees C."
        )
        return result

    has_claim = "claimed_state" in seen
    last_timestamp: datetime | None = None

    for row_number, row in enumerate(reader, start=1):
        result.total_rows += 1
        if not any(cell.strip() for cell in row):
            result.warnings.append(
                f"Row {row_number} is blank and was skipped."
            )
            result.total_rows -= 1
            continue

        def cell_for(canonical: str) -> str | None:
            for idx, name in header_map.items():
                if name == canonical and idx < len(row):
                    return row[idx]
            return None

        ts_raw = cell_for("timestamp")
        timestamp, ts_error = _parse_timestamp(ts_raw)
        if ts_error:
            result.errors.append(f"Row {row_number}: {ts_error}.")
        elif (
            timestamp is not None
            and last_timestamp is not None
            and timestamp < last_timestamp
        ):
            result.warnings.append(
                f"Row {row_number}: timestamp is earlier than the previous "
                "row (out of order)."
            )
        if timestamp is not None:
            last_timestamp = timestamp

        observed: dict[str, float | None] = {}
        quality: dict[str, str] = {}
        for key in MEASUREMENT_KEYS:
            value, flag, error = _parse_value(key, cell_for(key))
            observed[key] = value
            quality[key] = flag
            if error:
                result.errors.append(f"Row {row_number}: {error}.")

        claim_raw = cell_for("claimed_state") if has_claim else None
        claimed_state = (
            str(claim_raw).strip()
            if claim_raw is not None and str(claim_raw).strip()
            else default_claimed_state
        )

        result.records.append(
            CSVRecord(
                index=len(result.records),
                row_number=row_number,
                timestamp=timestamp,
                observed=observed,
                quality=quality,
                claimed_state=claimed_state,
            )
        )

    if not result.records:
        result.errors.append("No data rows were found in the file.")
    return result


def format_summary(result: CsvLoadResult) -> str:
    """One-line human summary of a parse result."""
    if not result.records:
        return "No usable rows loaded."
    return (
        f"{len(result.records)} row(s) loaded from "
        f"{result.total_rows} total row(s)."
    )
