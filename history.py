"""Event-history helpers for the Chemical Truth Engine demo dashboard.

This small module keeps the demonstration history logic (records, copying,
tabular export, trend extraction) free of Streamlit imports so it can be unit
tested directly. The dashboard stores history in ``st.session_state``, which
is TEMPORARY: it disappears when the browser tab is closed or Streamlit
restarts. Nothing is written to disk or sent anywhere.
"""

from __future__ import annotations

import copy
from datetime import datetime
from math import isfinite
from typing import Any, Mapping

import pandas as pd

from detection_engine import MEASUREMENT_KEYS, VerificationResult

__all__ = [
    "HISTORY_COLUMNS",
    "append_record",
    "build_record",
    "clear_history",
    "next_run_number",
    "to_dataframe",
    "trend_dataframe",
]

#: Column order used for the history table in the dashboard.
HISTORY_COLUMNS = [
    "Run",
    "Timestamp (local time)",
    "Source",
    "Reference",
    "Scenario",
    "Result",
    "Explanation",
]


def _optional_float(value: Any) -> float | None:
    """Return a finite float, or None for anything unusable (None, NaN,
    inf, strings, booleans, ...)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) else None


def build_record(
    *,
    run: int,
    timestamp: datetime,
    scenario_id: str,
    scenario_name: str,
    claimed_state: str,
    observed: Any,
    expected: Mapping[str, float],
    tolerances: Mapping[str, float],
    result: VerificationResult,
    source: str = "SIMULATION",
    reference_source: str | None = None,
    sample_time: str | None = None,
) -> dict[str, Any]:
    """Build one immutable-by-convention history record.

    ``timestamp`` is converted to the computer's local time zone and stored as
    a readable string. ``source`` labels where the readings came from
    (SIMULATION, MANUAL INPUT, CSV REPLAY or SERIAL). ``reference_source``
    labels where the expected values / tolerances came from when that was an
    explicit choice (manual entry); it is None when the Configuration-page
    values were used implicitly. ``sample_time`` is the timestamp of
    the measurement itself when it differs from the verification time.
    Non-finite observed values (NaN/inf) are normalised to ``None`` so that
    later charting and display never see them.
    """
    obs_map = observed if isinstance(observed, Mapping) else {}
    ts = timestamp.astimezone()
    return {
        "run": int(run),
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source": source,
        "reference_source": reference_source,
        "sample_time": sample_time,
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "claimed_state": claimed_state,
        "result": result.status.value,
        "summary": result.summary,
        "reasons": tuple(result.reasons),
        "observed": {
            key: _optional_float(obs_map.get(key)) for key in MEASUREMENT_KEYS
        },
        "expected": {
            key: _optional_float(expected.get(key)) for key in MEASUREMENT_KEYS
        },
        "tolerances": {
            key: _optional_float(tolerances.get(key)) for key in MEASUREMENT_KEYS
        },
        "residuals": {check.key: check.residual for check in result.checks},
        "passed": {check.key: check.within_tolerance for check in result.checks},
    }


def next_run_number(history: list[dict[str, Any]]) -> int:
    """The run number for the next record (1 for an empty history)."""
    return len(history) + 1


def append_record(
    history: list[dict[str, Any]],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Append an independent snapshot of ``record``.

    The stored entry is a deep copy, so later mutation of the caller's record
    (or of one history entry) can never corrupt the others.

    Returns the stored snapshot itself.
    """
    stored = copy.deepcopy(dict(record))
    history.append(stored)
    return stored


def clear_history(history: list[dict[str, Any]]) -> None:
    """Empty the demonstration history in place."""
    history.clear()


def to_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert history records into the dashboard table.

    An empty history yields an empty DataFrame with the expected columns
    rather than raising an exception.
    """
    if not history:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    rows = [
        {
            "Run": record.get("run"),
            "Timestamp (local time)": record.get("timestamp"),
            "Source": record.get("source", "SIMULATION"),
            "Reference": record.get("reference_source") or "-",
            "Scenario": record.get("scenario_name"),
            "Result": record.get("result"),
            "Explanation": record.get("summary"),
        }
        for record in history
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def trend_dataframe(
    history: list[dict[str, Any]],
    key: str,
) -> pd.DataFrame:
    """Expected vs observed values for one measurement across all runs.

    Used by the trend charts. Missing readings appear as empty cells.
    """
    columns = ["Run", "Expected", "Observed"]
    if not history:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "Run": record.get("run"),
            "Expected": record.get("expected", {}).get(key),
            "Observed": record.get("observed", {}).get(key),
        }
        for record in history
    ]
    return pd.DataFrame(rows, columns=columns)
