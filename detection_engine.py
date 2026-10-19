"""Detection engine for Chemical Truth Engine 2.0.

Physics-informed integrity verification logic. This module is intentionally
free of any Streamlit (or other UI) dependency so it can be imported and unit
tested on its own.

Definitions
-----------
* expected   -- expected measurement for the controller's claimed state
                (a synthetic demo configuration in this prototype).
* observed   -- the measurement reported by the (simulated) sensor.
* tolerance  -- maximum acceptable absolute difference for that measurement.
* residual   -- observed - expected, expressed in the measurement's own unit.

Tolerance rule
--------------
A measurement PASSES when abs(residual) <= tolerance (inclusive).
A measurement FAILS when abs(residual) > tolerance.
Missing or invalid required data NEVER passes silently: it produces
INSUFFICIENT DATA.

Exact result labels produced by this engine:
    NORMAL | INTEGRITY ANOMALY | INSUFFICIENT DATA

An anomaly is a numerical discrepancy only. It is not proof of a cyberattack;
sensor faults, noise, calibration errors, environmental variation or
communication problems can produce the same observation. This engine does not
attempt to identify an attacker, their intent or their capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Any, Mapping

__all__ = [
    "DECIMALS",
    "LABELS",
    "MEASUREMENT_KEYS",
    "PLAUSIBLE_RANGES",
    "UNIT_SUFFIX",
    "UNITS",
    "MeasurementCheck",
    "VerificationResult",
    "VerificationStatus",
    "validate_configuration",
    "validate_observed",
    "verify",
]

# --------------------------------------------------------------------------- #
# Measurement metadata (units are consistent everywhere in the project)
# --------------------------------------------------------------------------- #

MEASUREMENT_KEYS: tuple[str, ...] = ("ph", "conductivity", "temperature")

LABELS: dict[str, str] = {
    "ph": "pH",
    "conductivity": "Conductivity",
    "temperature": "Temperature",
}

#: Display unit for each measurement. pH is dimensionless.
UNITS: dict[str, str] = {
    "ph": "dimensionless",
    "conductivity": "\u00b5S/cm",
    "temperature": "\u00b0C",
}

#: Suffix used inside reason sentences ("" for the dimensionless pH value).
UNIT_SUFFIX: dict[str, str] = {
    "ph": "",
    "conductivity": " \u00b5S/cm",
    "temperature": " \u00b0C",
}

#: Decimal places used when formatting each measurement for humans.
DECIMALS: dict[str, int] = {
    "ph": 2,
    "conductivity": 1,
    "temperature": 2,
}

#: Plausible physical ranges used to reject obviously invalid data.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "ph": (0.0, 14.0),
    "conductivity": (0.0, 100_000.0),
    "temperature": (-50.0, 200.0),
}


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

class VerificationStatus(str, Enum):
    """The three exact primary status labels used by the system."""

    NORMAL = "NORMAL"
    INTEGRITY_ANOMALY = "INTEGRITY ANOMALY"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class MeasurementCheck:
    """The comparison result for a single measurement."""

    key: str
    label: str
    unit: str
    expected: float | None
    observed: float | None
    tolerance: float | None
    residual: float | None
    absolute_residual: float | None
    within_tolerance: bool | None
    exceeded_by: float | None
    valid: bool
    issue: str | None

    @property
    def passed(self) -> bool | None:
        """True when the measurement compared and passed, False when it failed,
        None when it could not be compared at all."""
        return self.within_tolerance


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of one verification run."""

    status: VerificationStatus
    claimed_state: str
    checks: tuple[MeasurementCheck, ...]
    reasons: tuple[str, ...]
    summary: str

    @property
    def status_label(self) -> str:
        return self.status.value


# --------------------------------------------------------------------------- #
# Low-level validation helpers
# --------------------------------------------------------------------------- #

def _is_finite_number(value: Any) -> bool:
    """True only for finite real numbers (booleans and NaN/inf rejected)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return isfinite(float(value))
    except (TypeError, ValueError, OverflowError):  # pragma: no cover
        return False


def _in_plausible_range(key: str, value: float) -> bool:
    low, high = PLAUSIBLE_RANGES[key]
    return low <= value <= high


def _fmt(value: float | None, key: str) -> str:
    """Format a value for display inside reason sentences."""
    if value is None:
        return "-"
    return f"{value:.{DECIMALS[key]}f}"


def _lookup(data: Any, key: str) -> Any:
    """Fetch a key from a mapping without ever raising."""
    if isinstance(data, Mapping):
        return data.get(key, None)
    return None


def _claim_issues(claimed_state: Any) -> list[str]:
    if not isinstance(claimed_state, str) or not claimed_state.strip():
        return [
            "A non-empty controller-claimed state is required for verification."
        ]
    return []


#: Data-quality flags that are acceptable for a required measurement.
_ACCEPTABLE_QUALITIES = frozenset({"ok"})


def _quality_issues(data_quality: Any) -> list[str]:
    """Validate per-measurement data-quality flags.

    Only ``ok`` is acceptable for a required measurement. ``stale``,
    ``missing``, ``invalid`` and any unknown flag are blocking, because a
    measurement whose freshness or validity is in doubt must never count as
    a pass.
    """
    if data_quality is None:
        return []
    if not isinstance(data_quality, Mapping):
        return [
            "Data-quality flags must be a mapping of measurement names to "
            f"flags (received {type(data_quality).__name__})."
        ]
    issues: list[str] = []
    for key in MEASUREMENT_KEYS:
        if key not in data_quality:
            continue  # absent flag is unknown, not a failure
        flag = data_quality[key]
        label = LABELS[key]
        if not isinstance(flag, str):
            issues.append(
                f"Data-quality flag for {label} must be a string "
                f"(received {flag!r})."
            )
            continue
        normalized = flag.strip().lower()
        if normalized in _ACCEPTABLE_QUALITIES:
            continue
        if normalized == "stale":
            issues.append(
                f"Observed {label} is stale: the newest value is too old to "
                "compare against the expected response."
            )
        elif normalized == "missing":
            issues.append(f"Observed {label} is flagged as missing.")
        elif normalized == "invalid":
            issues.append(f"Observed {label} is flagged as invalid.")
        else:
            issues.append(
                f"Observed {label} has an unknown data-quality flag "
                f"({flag!r})."
            )
    return issues


def _freshness_issues(
    sample_age_seconds: Any,
    max_age_seconds: Any,
) -> list[str]:
    """Validate sample freshness against the configured limit."""
    if sample_age_seconds is None and max_age_seconds is None:
        return []
    if sample_age_seconds is None or max_age_seconds is None:
        return [
            "sample_age_seconds and max_age_seconds must be provided "
            "together to check freshness."
        ]
    if not _is_finite_number(sample_age_seconds):
        return [
            "sample_age_seconds must be a finite number of seconds "
            f"(received {sample_age_seconds!r})."
        ]
    if not _is_finite_number(max_age_seconds):
        return [
            "max_age_seconds must be a finite number of seconds "
            f"(received {max_age_seconds!r})."
        ]
    age = float(sample_age_seconds)
    limit = float(max_age_seconds)
    if age < 0:
        return [
            "sample_age_seconds must not be negative "
            f"(received {sample_age_seconds})."
        ]
    if limit < 0:
        return [
            "max_age_seconds must not be negative "
            f"(received {max_age_seconds})."
        ]
    if age > limit:
        return [
            f"The newest sample is {age:.1f} s old, which exceeds the "
            f"freshness limit of {limit:.1f} s. The data is stale, so no "
            "conclusion is drawn."
        ]
    return []


def _expected_issues(key: str, expected: Any) -> list[str]:
    label = LABELS[key]
    low, high = PLAUSIBLE_RANGES[key]
    unit = UNITS[key]
    if not isinstance(expected, Mapping) or key not in expected:
        return [f"Expected {label} is missing."]
    raw = expected[key]
    if not _is_finite_number(raw):
        return [
            f"Expected {label} must be a finite number "
            f"(received {raw!r})."
        ]
    value = float(raw)
    if not _in_plausible_range(key, value):
        return [
            f"Expected {label} value {value} is outside the plausible "
            f"range {low} to {high} {unit}."
        ]
    return []


def _tolerance_issues(key: str, tolerances: Any) -> list[str]:
    label = LABELS[key]
    if not isinstance(tolerances, Mapping) or key not in tolerances:
        return [f"Tolerance for {label} is missing."]
    raw = tolerances[key]
    if not _is_finite_number(raw):
        return [
            f"Tolerance for {label} must be a finite number "
            f"(received {raw!r})."
        ]
    if float(raw) < 0:
        return [
            f"Tolerance for {label} must not be negative "
            f"(received {raw})."
        ]
    return []


def _observed_issues(key: str, observed: Any) -> list[str]:
    label = LABELS[key]
    low, high = PLAUSIBLE_RANGES[key]
    unit = UNITS[key]
    if not isinstance(observed, Mapping) or key not in observed:
        return [f"Observed {label} is missing."]
    raw = observed[key]
    if raw is None:
        return [f"Observed {label} is missing (None)."]
    if not _is_finite_number(raw):
        return [
            f"Observed {label} must be a finite number "
            f"(received {raw!r})."
        ]
    value = float(raw)
    if not _in_plausible_range(key, value):
        return [
            f"Observed {label} value {value} is outside the plausible "
            f"range {low} to {high} {unit}."
        ]
    return []


def validate_configuration(
    expected: Any,
    tolerances: Any,
) -> list[str]:
    """Validate expected values and tolerances.

    Returns a list of human-readable problems. An empty list means the
    configuration is valid. No value is ever silently substituted.
    """
    issues: list[str] = []
    if not isinstance(expected, Mapping):
        issues.append(
            "Expected values must be a mapping of measurement names to "
            f"numbers (received {type(expected).__name__})."
        )
    if not isinstance(tolerances, Mapping):
        issues.append(
            "Tolerances must be a mapping of measurement names to numbers "
            f"(received {type(tolerances).__name__})."
        )
    for key in MEASUREMENT_KEYS:
        if isinstance(expected, Mapping):
            issues.extend(_expected_issues(key, expected))
        if isinstance(tolerances, Mapping):
            issues.extend(_tolerance_issues(key, tolerances))
    return issues


def validate_observed(observed: Any) -> list[str]:
    """Validate the observed readings.

    Returns a list of human-readable problems. An empty list means every
    required measurement is present, numeric, finite and plausible.
    """
    if not isinstance(observed, Mapping):
        return [
            "Observed readings must be a mapping of measurement names to "
            f"numbers (received {type(observed).__name__})."
        ]
    issues: list[str] = []
    for key in MEASUREMENT_KEYS:
        issues.extend(_observed_issues(key, observed))
    return issues


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def verify(
    observed: Any,
    expected: Any,
    tolerances: Any,
    claimed_state: Any,
    *,
    data_quality: Any = None,
    sample_age_seconds: Any = None,
    max_age_seconds: Any = None,
) -> VerificationResult:
    """Compare observed readings against the expected response.

    Parameters
    ----------
    observed:
        Mapping of measurement key -> observed value (``ph``,
        ``conductivity``, ``temperature``). Values may be missing/invalid;
        such inputs yield INSUFFICIENT DATA rather than a crash.
    expected:
        Mapping of measurement key -> expected value for the claimed state.
    tolerances:
        Mapping of measurement key -> non-negative tolerance.
    claimed_state:
        Non-empty string describing the state the controller claims.
    data_quality:
        Optional mapping of measurement key -> data-quality flag
        (``ok``, ``stale``, ``missing``, ``invalid``). A required measurement
        whose flag is not ``ok`` cannot pass; it forces INSUFFICIENT DATA.
        A key that is absent from the mapping is treated as unknown and
        imposes no restriction.
    sample_age_seconds:
        Optional age of the newest sample, in seconds. Used for freshness.
    max_age_seconds:
        Optional freshness limit in seconds. If ``sample_age_seconds``
        exceeds it, the data is stale and the result is INSUFFICIENT DATA.

    Returns
    -------
    VerificationResult
        ``status`` is one of NORMAL, INTEGRITY ANOMALY, INSUFFICIENT DATA,
        accompanied by per-measurement checks and structured reasons.

    Notes
    -----
    * residual = observed - expected (per measurement, in its own unit).
    * abs(residual) <= tolerance  -> pass;  abs(residual) > tolerance -> fail.
    * Missing, NaN, infinite, non-numeric, out-of-range, stale or otherwise
      invalid required data can never pass; it produces INSUFFICIENT DATA.
    * The engine reports *what* disagrees, never *why* or *who* caused it.
    """
    config_issues = validate_configuration(expected, tolerances)
    observed_issues = validate_observed(observed)
    claim_issues = _claim_issues(claimed_state)
    quality_issues = _quality_issues(data_quality)
    freshness_issues = _freshness_issues(sample_age_seconds, max_age_seconds)

    checks: list[MeasurementCheck] = []
    for key in MEASUREMENT_KEYS:
        exp_problems = _expected_issues(key, expected)
        tol_problems = _tolerance_issues(key, tolerances)
        obs_problems = _observed_issues(key, observed)
        problems = exp_problems + tol_problems + obs_problems

        exp_raw = _lookup(expected, key)
        obs_raw = _lookup(observed, key)
        tol_raw = _lookup(tolerances, key)

        exp_value = float(exp_raw) if _is_finite_number(exp_raw) else None
        obs_value = float(obs_raw) if _is_finite_number(obs_raw) else None
        tol_value = float(tol_raw) if _is_finite_number(tol_raw) else None

        expected_ok = exp_value is not None and _in_plausible_range(key, exp_value)
        observed_ok = obs_value is not None and _in_plausible_range(key, obs_value)
        tolerance_ok = tol_value is not None and tol_value >= 0.0

        residual: float | None = None
        absolute: float | None = None
        within: bool | None = None
        exceeded_by: float | None = 0.0

        if expected_ok and observed_ok:
            residual = obs_value - exp_value
            absolute = abs(residual)
            if tolerance_ok:
                # Inclusive comparison: equal to the tolerance passes.
                within = absolute <= tol_value
                exceeded_by = 0.0 if within else absolute - tol_value

        valid = expected_ok and observed_ok and tolerance_ok
        checks.append(
            MeasurementCheck(
                key=key,
                label=LABELS[key],
                unit=UNITS[key],
                expected=exp_value,
                observed=obs_value,
                tolerance=tol_value,
                residual=residual,
                absolute_residual=absolute,
                within_tolerance=within,
                exceeded_by=exceeded_by if within is not None else None,
                valid=valid,
                issue="; ".join(problems) if problems else None,
            )
        )

    blocking = (
        config_issues
        + observed_issues
        + claim_issues
        + quality_issues
        + freshness_issues
    )
    if blocking:
        reasons = list(blocking)
        # Informational only: some valid measurements may already be beyond
        # tolerance, but a conclusion is withheld because required data is
        # invalid or missing.
        already_failing = [
            c for c in checks if c.valid and c.within_tolerance is False
        ]
        for check in already_failing:
            reasons.append(
                f"Informational: {check.label} also exceeds its tolerance "
                f"(residual {_fmt(check.residual, check.key)} vs "
                f"+/-{_fmt(check.tolerance, check.key)}"
                f"{UNIT_SUFFIX[check.key]}), but no conclusion is drawn "
                "while required data is invalid."
            )
        reasons.append(
            "Required information was missing or invalid, so the check was "
            "not completed. Missing or invalid data never counts as a pass, "
            "and it is not automatically an attack either."
        )
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_DATA,
            claimed_state=claimed_state if isinstance(claimed_state, str) else "",
            checks=tuple(checks),
            reasons=tuple(reasons),
            summary=(
                "Required information was missing or invalid, so no "
                "verification conclusion could be drawn."
            ),
        )

    failing = [c for c in checks if c.within_tolerance is False]
    if failing:
        reasons = []
        for check in failing:
            reasons.append(
                f"{check.label}: observed {_fmt(check.observed, check.key)} "
                f"vs expected {_fmt(check.expected, check.key)}"
                f"{UNIT_SUFFIX[check.key]} -> residual "
                f"{check.residual:+.{DECIMALS[check.key]}f} exceeds the "
                f"tolerance +/-{_fmt(check.tolerance, check.key)}"
                f"{UNIT_SUFFIX[check.key]} by "
                f"{check.exceeded_by:.{DECIMALS[check.key]}f}"
                f"{UNIT_SUFFIX[check.key]}."
            )
        reasons.append(
            "This is a numerical discrepancy between observation and "
            "expected behaviour. It does not identify a cause: sensor "
            "faults, noise, calibration errors, environment or manipulated "
            "data can all look the same."
        )
        return VerificationResult(
            status=VerificationStatus.INTEGRITY_ANOMALY,
            claimed_state=claimed_state,
            checks=tuple(checks),
            reasons=tuple(reasons),
            summary=(
                "At least one measurement deviates from the expected "
                "response beyond its configured tolerance. This flags an "
                "integrity discrepancy; it does not by itself prove a "
                "cyberattack."
            ),
        )

    reasons = [
        f"{check.label}: |residual| "
        f"{check.absolute_residual:.{DECIMALS[check.key]}f} <= tolerance "
        f"+/-{_fmt(check.tolerance, check.key)}{UNIT_SUFFIX[check.key]} "
        f"(observed {_fmt(check.observed, check.key)} vs expected "
        f"{_fmt(check.expected, check.key)})."
        for check in checks
    ]
    reasons.append(
        "Every required measurement agreed with the expected response "
        "within its configured tolerance."
    )
    return VerificationResult(
        status=VerificationStatus.NORMAL,
        claimed_state=claimed_state,
        checks=tuple(checks),
        reasons=tuple(reasons),
        summary=(
            "Observed measurements agree with the expected response for the "
            "claimed state within the configured tolerances."
        ),
    )
