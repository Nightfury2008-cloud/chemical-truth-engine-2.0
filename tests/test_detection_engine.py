"""Unit tests for detection_engine.verify() and its validation helpers.

These tests exercise real behaviour (classification, residual maths, and
graceful rejection of bad input) rather than restating the source code.
"""

import math

import pytest

from detection_engine import (
    VerificationStatus,
    validate_configuration,
    validate_observed,
    verify,
)

# Synthetic demo defaults (mirror of simulation.DEFAULT_*).
EXPECTED = {"ph": 8.0, "conductivity": 650.0, "temperature": 25.5}
TOLERANCES = {"ph": 0.5, "conductivity": 80.0, "temperature": 2.0}


def _observed(**overrides):
    """Valid readings comfortably inside every default tolerance."""
    base = {"ph": 8.1, "conductivity": 640.0, "temperature": 25.2}
    base.update(overrides)
    return base


def _check(result, key):
    return next(c for c in result.checks if c.key == key)


# --------------------------------------------------------------------------- #
# 1. Classification: NORMAL
# --------------------------------------------------------------------------- #

def test_all_within_tolerance_is_normal():
    result = verify(_observed(), EXPECTED, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.NORMAL
    assert result.status_label == "NORMAL"
    for check in result.checks:
        assert check.valid is True
        assert check.within_tolerance is True
        assert check.issue is None


def test_normal_result_has_summary_and_reasons():
    result = verify(_observed(), EXPECTED, TOLERANCES, "Steady operation")
    assert result.summary
    assert any("within" in r.lower() for r in result.reasons)


# --------------------------------------------------------------------------- #
# 2-4. Classification: INTEGRITY ANOMALY per measurement
# --------------------------------------------------------------------------- #

def test_ph_beyond_tolerance_is_anomaly():
    result = verify(_observed(ph=9.5), EXPECTED, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY
    check = _check(result, "ph")
    assert check.within_tolerance is False
    # The other two measurements are still fine.
    assert _check(result, "conductivity").within_tolerance is True
    assert _check(result, "temperature").within_tolerance is True


def test_conductivity_beyond_tolerance_is_anomaly():
    result = verify(
        _observed(conductivity=800.0), EXPECTED, TOLERANCES, "Steady operation"
    )
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY
    assert _check(result, "conductivity").within_tolerance is False


def test_temperature_beyond_tolerance_is_anomaly():
    result = verify(
        _observed(temperature=30.0), EXPECTED, TOLERANCES, "Steady operation"
    )
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY
    assert _check(result, "temperature").within_tolerance is False


def test_multiple_failures_all_reported():
    result = verify(
        _observed(ph=9.5, temperature=30.0),
        EXPECTED,
        TOLERANCES,
        "Steady operation",
    )
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY
    failed = {c.key for c in result.checks if c.within_tolerance is False}
    assert failed == {"ph", "temperature"}


# --------------------------------------------------------------------------- #
# 5. Boundary rule: |residual| <= tolerance passes, > tolerance fails
# --------------------------------------------------------------------------- #

def test_residual_equal_to_tolerance_passes():
    # pH 8.5 - 8.0 == 0.5, exactly the tolerance -> must PASS.
    result = verify(_observed(ph=8.5), EXPECTED, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.NORMAL


def test_residual_just_past_tolerance_fails():
    result = verify(
        _observed(ph=8.50001), EXPECTED, TOLERANCES, "Steady operation"
    )
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY


def test_residual_exactly_tolerance_for_each_measurement():
    cases = {
        "ph": 8.5,             # 8.5 - 8.0 == 0.5
        "conductivity": 730.0,  # 730 - 650 == 80
        "temperature": 27.5,    # 27.5 - 25.5 == 2.0
    }
    for key, value in cases.items():
        result = verify(
            _observed(**{key: value}), EXPECTED, TOLERANCES, "Steady operation"
        )
        assert result.status is VerificationStatus.NORMAL, key


# --------------------------------------------------------------------------- #
# Residual maths
# --------------------------------------------------------------------------- #

def test_residual_is_observed_minus_expected_with_correct_sign():
    result = verify(_observed(ph=8.3), EXPECTED, TOLERANCES, "Steady operation")
    assert _check(result, "ph").residual == pytest.approx(0.3)
    result = verify(_observed(ph=7.0), EXPECTED, TOLERANCES, "Steady operation")
    assert _check(result, "ph").residual == pytest.approx(-1.0)


def test_conductivity_residual_magnitude():
    result = verify(
        _observed(conductivity=750.0), EXPECTED, TOLERANCES, "Steady operation"
    )
    check = _check(result, "conductivity")
    assert check.residual == pytest.approx(100.0)
    assert check.exceeded_by == pytest.approx(20.0)  # 100 - 80


# --------------------------------------------------------------------------- #
# 6-8. INSUFFICIENT DATA: missing, NaN, infinite, non-numeric
# --------------------------------------------------------------------------- #

def test_missing_observed_key_is_insufficient():
    observed = {"ph": 8.1, "conductivity": 640.0}  # temperature absent
    result = verify(observed, EXPECTED, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_none_observed_value_is_insufficient():
    result = verify(
        _observed(ph=None), EXPECTED, TOLERANCES, "Steady operation"
    )
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_nan_observed_is_insufficient():
    result = verify(
        _observed(ph=float("nan")), EXPECTED, TOLERANCES, "Steady operation"
    )
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_infinite_observed_is_insufficient():
    for bad in (float("inf"), float("-inf")):
        result = verify(
            _observed(temperature=bad), EXPECTED, TOLERANCES, "Steady operation"
        )
        assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_nonnumeric_observed_is_insufficient():
    for bad in ("warm", [25.5], {"value": 25.5}, object()):
        result = verify(
            _observed(temperature=bad), EXPECTED, TOLERANCES, "Steady operation"
        )
        assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_bool_observed_is_insufficient():
    # bool is a subclass of int in Python; it must not be accepted as a number.
    result = verify(_observed(ph=True), EXPECTED, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_invalid_data_never_passes_even_if_other_values_fail_tolerance():
    # temperature is out of tolerance, but pH is NaN -> no conclusion.
    result = verify(
        _observed(ph=float("nan"), temperature=30.0),
        EXPECTED,
        TOLERANCES,
        "Steady operation",
    )
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# 9-10. Invalid configuration
# --------------------------------------------------------------------------- #

def test_negative_tolerance_is_rejected():
    bad = dict(TOLERANCES, ph=-0.1)
    result = verify(_observed(), EXPECTED, bad, "Steady operation")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_nan_and_infinite_tolerances_are_rejected():
    for bad_value in (float("nan"), float("inf")):
        bad = dict(TOLERANCES, conductivity=bad_value)
        result = verify(_observed(), EXPECTED, bad, "Steady operation")
        assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_zero_tolerance_is_allowed_and_requires_exact_match():
    zero = {"ph": 0.0, "conductivity": 0.0, "temperature": 0.0}
    exact = {"ph": 8.0, "conductivity": 650.0, "temperature": 25.5}
    assert verify(exact, EXPECTED, zero, "Steady").status is VerificationStatus.NORMAL
    off = {"ph": 8.001, "conductivity": 650.0, "temperature": 25.5}
    assert (
        verify(off, EXPECTED, zero, "Steady").status
        is VerificationStatus.INTEGRITY_ANOMALY
    )


def test_nan_expected_value_is_rejected():
    bad = dict(EXPECTED, ph=float("nan"))
    result = verify(_observed(), bad, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_out_of_plausible_range_expected_is_rejected():
    # pH 99 is not physically meaningful.
    bad = dict(EXPECTED, ph=99.0)
    result = verify(_observed(), bad, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_out_of_plausible_range_observed_is_rejected():
    result = verify(
        _observed(ph=99.0), EXPECTED, TOLERANCES, "Steady operation"
    )
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_nonnumeric_expected_is_rejected():
    bad = dict(EXPECTED, temperature="warm")
    result = verify(_observed(), bad, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_missing_expected_or_tolerance_key_is_rejected():
    partial_expected = {"ph": 8.0, "conductivity": 650.0}
    result = verify(_observed(), partial_expected, TOLERANCES, "Steady")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA

    partial_tolerances = {"ph": 0.5, "conductivity": 80.0}
    result = verify(_observed(), EXPECTED, partial_tolerances, "Steady")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# 11. Empty / malformed inputs never crash
# --------------------------------------------------------------------------- #

def test_empty_mappings_do_not_crash():
    result = verify({}, {}, {}, "Steady operation")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_none_inputs_do_not_crash():
    result = verify(None, None, None, None)
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_wrong_container_types_do_not_crash():
    result = verify([1, 2, 3], "nope", 42, 99)
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_empty_or_whitespace_claim_is_insufficient():
    result = verify(_observed(), EXPECTED, TOLERANCES, "")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA
    result = verify(_observed(), EXPECTED, TOLERANCES, "   ")
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


def test_none_claim_is_insufficient():
    result = verify(_observed(), EXPECTED, TOLERANCES, None)
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# 17. Changing tolerances changes the result
# --------------------------------------------------------------------------- #

def test_widening_tolerance_turns_anomaly_into_normal():
    result = verify(_observed(ph=9.0), EXPECTED, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY

    wider = dict(TOLERANCES, ph=1.0)  # |9.0-8.0| == 1.0 <= 1.0
    result = verify(_observed(ph=9.0), EXPECTED, wider, "Steady operation")
    assert result.status is VerificationStatus.NORMAL


def test_narrowing_tolerance_turns_normal_into_anomaly():
    result = verify(_observed(), EXPECTED, TOLERANCES, "Steady operation")
    assert result.status is VerificationStatus.NORMAL

    narrower = dict(TOLERANCES, ph=0.05)  # |8.1-8.0| == 0.1 > 0.05
    result = verify(_observed(), EXPECTED, narrower, "Steady operation")
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY


# --------------------------------------------------------------------------- #
# Structured reasons
# --------------------------------------------------------------------------- #

def test_anomaly_reason_names_the_measurement_and_exceedance():
    result = verify(_observed(ph=9.5), EXPECTED, TOLERANCES, "Steady operation")
    joined = " ".join(result.reasons)
    assert "pH" in joined
    assert "1.0" in joined  # exceeded_by = 1.5 - 0.5


def test_anomaly_does_not_attribute_cause_or_attacker():
    result = verify(_observed(ph=9.5), EXPECTED, TOLERANCES, "Steady operation")
    joined = " ".join(result.reasons).lower() + result.summary.lower()
    for forbidden in ("attacker", "hack", "malware", "confirmed attack"):
        assert forbidden not in joined


def test_claimed_state_is_preserved_in_result():
    result = verify(_observed(), EXPECTED, TOLERANCES, "Target state reached")
    assert result.claimed_state == "Target state reached"


# --------------------------------------------------------------------------- #
# Validation helpers return useful, non-empty messages
# --------------------------------------------------------------------------- #

def test_validate_configuration_reports_every_missing_key():
    issues = validate_configuration({}, {})
    assert len(issues) >= 6  # 3 expected + 3 tolerances
    assert all(isinstance(i, str) and i for i in issues)


def test_validate_configuration_accepts_good_input():
    assert validate_configuration(EXPECTED, TOLERANCES) == []


def test_validate_observed_reports_missing_and_bad_values():
    issues = validate_observed({"ph": float("nan"), "conductivity": "x"})
    assert len(issues) >= 3
    assert all(isinstance(i, str) and i for i in issues)


def test_validate_observed_accepts_good_input():
    assert validate_observed(_observed()) == []


def test_validate_helpers_reject_non_mapping():
    assert validate_configuration("nope", 1)
    assert validate_observed(None)
    assert validate_observed(math.pi)
