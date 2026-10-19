"""Unit tests for csv_replay.parse_csv().

These exercise real validation behaviour: required columns, malformed rows,
missing/invalid values, timestamps and honesty about recorded data.
"""

from math import isnan

from csv_replay import REQUIRED_COLUMNS, parse_csv
from detection_engine import VerificationStatus, verify
from simulation import DEFAULT_EXPECTED, DEFAULT_TOLERANCES

GOOD_HEADER = "timestamp,ph,conductivity,temperature\n"


def _good_rows(n=3):
    lines = [GOOD_HEADER]
    for i in range(n):
        lines.append(f"2026-01-01T12:00:{i:02d},8.0{i},65{i},25.5{i}\n")
    return "".join(lines)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_required_columns_constant():
    assert "timestamp" in REQUIRED_COLUMNS
    assert "ph" in REQUIRED_COLUMNS
    assert "conductivity" in REQUIRED_COLUMNS
    assert "temperature" in REQUIRED_COLUMNS


def test_valid_file_parses_all_rows():
    result = parse_csv(_good_rows(4))
    assert result.ok
    assert len(result.records) == 4
    assert result.total_rows == 4
    assert result.errors == []


def test_parsed_values_are_floats_and_timestamps():
    result = parse_csv(_good_rows(1))
    record = result.records[0]
    assert record.timestamp is not None
    assert record.timestamp.year == 2026
    assert isinstance(record.observed["ph"], float)
    assert record.quality["ph"] == "ok"
    assert record.row_number == 1


def test_valid_rows_verify_normal_under_defaults():
    result = parse_csv(_good_rows(3))
    for record in result.records:
        r = verify(
            record.observed,
            DEFAULT_EXPECTED,
            DEFAULT_TOLERANCES,
            record.claimed_state,
            data_quality=record.quality,
        )
        assert r.status is VerificationStatus.NORMAL


def test_header_is_case_and_space_insensitive():
    text = "Timestamp, PH ,Conductivity,Temperature\n2026-01-01T12:00:00,8.0,650,25.5\n"
    result = parse_csv(text)
    assert result.ok


def test_claimed_state_column_is_used():
    text = (
        "timestamp,ph,conductivity,temperature,claimed_state\n"
        "2026-01-01T12:00:00,8.0,650,25.5,Target state reached\n"
    )
    result = parse_csv(text)
    assert result.ok
    assert result.records[0].claimed_state == "Target state reached"


def test_default_claimed_state_used_when_column_absent():
    result = parse_csv(_good_rows(1))
    assert result.records[0].claimed_state == "Steady operation"


def test_blank_claimed_state_falls_back_to_default():
    text = (
        "timestamp,ph,conductivity,temperature,claimed_state\n"
        "2026-01-01T12:00:00,8.0,650,25.5,\n"
    )
    result = parse_csv(text)
    assert result.records[0].claimed_state == "Steady operation"


def test_extra_columns_are_ignored():
    text = (
        "timestamp,ph,conductivity,temperature,operator_note\n"
        "2026-01-01T12:00:00,8.0,650,25.5,hello\n"
    )
    result = parse_csv(text)
    assert result.ok


# --------------------------------------------------------------------------- #
# Malformed files
# --------------------------------------------------------------------------- #

def test_empty_file_is_rejected_not_crashed():
    for text in ("", "   \n  ", None):
        result = parse_csv(text)
        assert not result.ok
        assert result.errors


def test_missing_columns_are_reported():
    result = parse_csv("timestamp,ph\n2026-01-01T12:00:00,8.0\n")
    assert not result.ok
    joined = " ".join(result.errors)
    assert "conductivity" in joined
    assert "temperature" in joined


def test_header_only_file_has_no_records():
    result = parse_csv(GOOD_HEADER)
    assert not result.ok
    assert result.records == []


def test_garbage_file_does_not_crash():
    result = parse_csv("this is not,a csv file at all\njust text here\n")
    assert not result.ok


def test_nonnumeric_cells_are_invalid_not_normal():
    text = GOOD_HEADER + "2026-01-01T12:00:00,abc,650,25.5\n"
    result = parse_csv(text)
    record = result.records[0]
    assert record.observed["ph"] is None
    assert record.quality["ph"] == "invalid"
    assert any("not a number" in e for e in result.errors)


def test_empty_cells_are_missing_not_normal():
    text = GOOD_HEADER + "2026-01-01T12:00:00,,650,25.5\n"
    result = parse_csv(text)
    record = result.records[0]
    assert record.observed["ph"] is None
    assert record.quality["ph"] == "missing"


def test_nan_and_inf_cells_are_invalid():
    for bad in ("nan", "inf", "-inf", "NaN"):
        text = GOOD_HEADER + f"2026-01-01T12:00:00,{bad},650,25.5\n"
        result = parse_csv(text)
        assert result.records[0].quality["ph"] == "invalid", bad


def test_out_of_range_values_are_invalid():
    text = GOOD_HEADER + "2026-01-01T12:00:00,99,650,25.5\n"
    result = parse_csv(text)
    assert result.records[0].quality["ph"] == "invalid"
    assert any("plausible range" in e for e in result.errors)


def test_bad_timestamp_is_reported_but_row_kept():
    text = GOOD_HEADER + "not-a-date,8.0,650,25.5\n"
    result = parse_csv(text)
    assert len(result.records) == 1
    assert result.records[0].timestamp is None
    assert result.records[0].quality["ph"] == "ok"
    assert any("ISO-8601" in e for e in result.errors)


def test_invalid_rows_verify_insufficient_not_normal():
    text = GOOD_HEADER + "2026-01-01T12:00:00,abc,650,25.5\n"
    result = parse_csv(text)
    record = result.records[0]
    r = verify(
        record.observed,
        DEFAULT_EXPECTED,
        DEFAULT_TOLERANCES,
        record.claimed_state,
        data_quality=record.quality,
    )
    assert r.status is VerificationStatus.INSUFFICIENT_DATA


def test_blank_rows_are_skipped_with_warning():
    text = GOOD_HEADER + "2026-01-01T12:00:00,8.0,650,25.5\n,,,\n"
    result = parse_csv(text)
    assert len(result.records) == 1
    assert any("blank" in w.lower() for w in result.warnings)


def test_out_of_order_timestamps_warn():
    text = (
        GOOD_HEADER
        + "2026-01-01T12:01:00,8.0,650,25.5\n"
        + "2026-01-01T12:00:00,8.0,650,25.5\n"
    )
    result = parse_csv(text)
    assert any("earlier" in w for w in result.warnings)


def test_duplicate_columns_warn():
    text = (
        "timestamp,ph,ph,conductivity,temperature\n"
        "2026-01-01T12:00:00,8.0,8.1,650,25.5\n"
    )
    result = parse_csv(text)
    assert any("Duplicate" in w for w in result.warnings)
    assert result.records[0].observed["ph"] == 8.0


def test_iso_timestamp_with_timezone_is_normalised():
    text = GOOD_HEADER + "2026-01-01T12:00:00+02:00,8.0,650,25.5\n"
    result = parse_csv(text)
    record = result.records[0]
    assert record.timestamp is not None
    assert record.timestamp.tzinfo is None


def test_claim_mismatch_profile_row_verifies_anomaly():
    # Recorded row where the controller claims the target state but the
    # readings still look like the steady state.
    text = (
        "timestamp,ph,conductivity,temperature,claimed_state\n"
        "2026-01-01T12:00:00,8.0,650,25.5,Target state reached\n"
    )
    result = parse_csv(text)
    record = result.records[0]
    from process_profiles import expected_for_state

    expected = expected_for_state("target_reached")
    r = verify(
        record.observed,
        expected,
        DEFAULT_TOLERANCES,
        record.claimed_state,
        data_quality=record.quality,
    )
    assert r.status is VerificationStatus.INTEGRITY_ANOMALY
