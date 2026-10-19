"""Unit tests for the session-history helpers in history.py.

These cover requirement 16 (repeated verification must not corrupt earlier
records) and the empty-history behaviour that the dashboard relies on.
"""

from datetime import datetime

import pytest

from detection_engine import MEASUREMENT_KEYS, verify
from history import (
    HISTORY_COLUMNS,
    append_record,
    build_record,
    clear_history,
    next_run_number,
    to_dataframe,
    trend_dataframe,
)
from simulation import DEFAULT_EXPECTED, DEFAULT_TOLERANCES, generate_sample

EXPECTED = DEFAULT_EXPECTED
TOLERANCES = DEFAULT_TOLERANCES


def _record(run=1, scenario="normal", when=None):
    sample = generate_sample(scenario)
    result = verify(
        sample.observed, EXPECTED, TOLERANCES, sample.claimed_state
    )
    return build_record(
        run=run,
        timestamp=when or datetime(2026, 1, 1, 12, 0, 0),
        scenario_id=sample.scenario_id,
        scenario_name=sample.scenario_name,
        claimed_state=sample.claimed_state,
        observed=sample.observed,
        expected=EXPECTED,
        tolerances=TOLERANCES,
        result=result,
    )


def test_build_record_contains_all_dashboard_fields():
    record = _record()
    for field in (
        "run",
        "timestamp",
        "scenario_id",
        "scenario_name",
        "result",
        "summary",
        "reasons",
        "observed",
        "expected",
        "tolerances",
        "residuals",
        "passed",
    ):
        assert field in record
    for key in MEASUREMENT_KEYS:
        assert key in record["observed"]
        assert key in record["expected"]
        assert key in record["tolerances"]


def test_timestamp_is_rendered_as_readable_local_string():
    record = _record()
    # "YYYY-MM-DD HH:MM:SS" plus a zone suffix (possibly empty).
    assert len(record["timestamp"]) >= 19
    assert record["timestamp"][4] == "-"
    assert record["timestamp"][10] == " "


def test_nan_and_inf_observed_are_normalised_to_none():
    record = _record(scenario="insufficient")
    assert record["observed"]["ph"] is None
    assert record["observed"]["conductivity"] is None  # was NaN


# --------------------------------------------------------------------------- #
# 16. Repeated verification must not corrupt earlier records
# --------------------------------------------------------------------------- #

def test_repeated_verification_preserves_earlier_records():
    history = []
    summaries = []
    for index in range(1, 6):
        record = _record(run=next_run_number(history))
        stored = append_record(history, record)
        summaries.append(stored["summary"])

    assert len(history) == 5
    assert [r["run"] for r in history] == [1, 2, 3, 4, 5]
    # Mutating a later record must not disturb the stored earlier ones.
    history[-1]["summary"] = "corrupted"
    history[-1]["observed"]["ph"] = None
    for index in range(4):
        assert history[index]["summary"] == summaries[index]
        assert history[index]["observed"]["ph"] is not None


def test_append_stores_an_independent_snapshot():
    history = []
    record = _record(run=next_run_number(history))
    original_ph = record["observed"]["ph"]
    original_result = record["result"]
    stored = append_record(history, record)
    # Mutating the caller's record afterwards must not affect the snapshot.
    record["result"] = "MUTATED"
    record["observed"]["ph"] = 999.0
    assert history[0]["result"] == original_result
    assert history[0]["observed"]["ph"] == original_ph
    # append_record returns the stored snapshot itself.
    assert stored is history[0]
    assert stored["result"] == original_result


def test_next_run_number_increments_and_handles_empty():
    assert next_run_number([]) == 1
    history = [_record(run=1), _record(run=2)]
    assert next_run_number(history) == 3


def test_clear_history_empties_the_list():
    history = [_record(run=1)]
    clear_history(history)
    assert history == []
    assert next_run_number(history) == 1


# --------------------------------------------------------------------------- #
# Empty-history handling must never raise (dashboard requirement)
# --------------------------------------------------------------------------- #

def test_to_dataframe_on_empty_history_returns_columns_not_error():
    df = to_dataframe([])
    assert list(df.columns) == HISTORY_COLUMNS
    assert len(df) == 0


def test_to_dataframe_rows_match_history():
    history = []
    for _ in range(3):
        append_record(history, _record(run=next_run_number(history)))
    df = to_dataframe(history)
    assert len(df) == 3
    assert list(df.columns) == HISTORY_COLUMNS
    assert list(df["Run"]) == [1, 2, 3]
    assert df["Result"].tolist() == ["NORMAL"] * 3


def test_trend_dataframe_on_empty_history_returns_columns_not_error():
    df = trend_dataframe([], "ph")
    assert list(df.columns) == ["Run", "Expected", "Observed"]
    assert len(df) == 0


def test_trend_dataframe_follows_history():
    history = []
    append_record(history, _record(run=1, scenario="normal"))
    append_record(history, _record(run=2, scenario="sensor_mismatch"))
    df = trend_dataframe(history, "ph")
    assert list(df["Run"]) == [1, 2]
    assert df["Expected"].tolist() == [EXPECTED["ph"]] * 2
    # The first run is inside tolerance; the second was pushed beyond it.
    assert df["Observed"][0] != df["Observed"][1]


def test_trend_dataframe_handles_missing_measurement():
    record = _record(scenario="insufficient")
    history = [record]
    df = trend_dataframe(history, "ph")
    assert len(df) == 1
    assert df["Observed"][0] is None or df["Observed"][0] != df["Observed"][0]


def test_trend_dataframe_works_for_every_measurement():
    history = [_record(run=1)]
    for key in MEASUREMENT_KEYS:
        df = trend_dataframe(history, key)
        assert len(df) == 1
        assert df["Expected"][0] == pytest.approx(EXPECTED[key])
