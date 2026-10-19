"""Manual user-input validation for Chemical Truth Engine 2.0.

This module turns the values typed into the dashboard's MANUAL INPUT form
into validated dictionaries ready for ``detection_engine.verify()``. It
contains NO comparison logic of its own - the detection engine stays the
single authority on residuals, tolerances and status labels. Like the other
input modules it is free of any Streamlit import so it can be unit tested
directly.

Validation rules (mirroring the engine's plausible physical ranges):

* pH           - required, finite, 0 to 14, dimensionless.
* Conductivity - required, finite, 0 to 100000 uS/cm.
* Temperature  - required, finite, -50 to 200 degrees C (the same
                 sensor-specific limits the rest of the application enforces).
* Tolerances   - required, finite, >= 0 (zero means "no deviation allowed").
* The claim must be one of :data:`CLAIM_OPTIONS`.

Invalid input is reported as a human-readable error; it is NEVER silently
replaced with normal values, and a verification is never run on it.

Claim semantics
---------------
The controller-reported state is *supporting evidence*, never proof that the
physical process is normal:

* ``TARGET REACHED / NORMAL`` - the claim the engine compares most directly:
  the expected values entered for it are the expected response for that
  claimed state; if the observations violate it beyond tolerance the engine
  flags INTEGRITY ANOMALY.
* ``PROCESS IN PROGRESS`` / ``FAULT REPORTED`` / ``UNKNOWN`` - recorded and
  shown, but never automatically treated as an attack or as inconsistent
  just because they are not "normal". Only the numbers are compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from detection_engine import LABELS, MEASUREMENT_KEYS, PLAUSIBLE_RANGES, UNITS

__all__ = [
    "CLAIM_NOTES",
    "CLAIM_OPTIONS",
    "ManualEntry",
    "parse_manual_entry",
]

#: Controller-reported states offered by the manual-entry form.
CLAIM_OPTIONS: tuple[str, ...] = (
    "TARGET REACHED / NORMAL",
    "PROCESS IN PROGRESS",
    "FAULT REPORTED",
    "UNKNOWN",
)

#: Honest, per-claim explanation shown next to the claim selector.
CLAIM_NOTES: dict[str, str] = {
    "TARGET REACHED / NORMAL": (
        "The controller says the process reached its target. The expected "
        "values you enter are treated as the expected response for that "
        "claim; observations that violate them beyond tolerance are flagged "
        "as an integrity anomaly."
    ),
    "PROCESS IN PROGRESS": (
        "A transitional claim. It is recorded but not judged - only the "
        "numbers are compared against the expected values you entered."
    ),
    "FAULT REPORTED": (
        "The controller itself reports a fault. This is supporting context, "
        "not a verdict: the engine still only compares numbers."
    ),
    "UNKNOWN": (
        "The claim is unknown. An unknown claim never counts as normal and "
        "is never automatically treated as an attack."
    ),
}


@dataclass(frozen=True)
class ManualEntry:
    """Validated manual input, ready for ``detection_engine.verify()``."""

    claimed_state: str
    observed: dict[str, float]
    expected: dict[str, float]
    tolerances: dict[str, float]


def _coerce_number_text(text: str) -> float | str | None:
    """Turn a text-field value into ``None`` / a float, or the text as-is.

    The dashboard's manual fields are plain text inputs (see
    ``app._panel_manual``), so values arrive as strings. Blank (or
    whitespace-only) text means the field was left empty and becomes
    ``None`` so it is reported as "is required". Text that parses as a
    number becomes that number. Anything else is returned unchanged so the
    caller reports ``must be a number (received ...)`` with exactly what
    the user typed - a malformed entry is never silently substituted.
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return text


def _check_number(
    value: Any,
    *,
    label: str,
    unit: str,
    low: float,
    high: float | None,
) -> tuple[float | None, str | None]:
    """Validate one typed number.

    Returns ``(value, error)``; exactly one is None. ``high=None`` means the
    value is unbounded above (used for tolerances). Invalid input is
    reported, never coerced into a normal-looking value.

    Values arriving from the text-input fields are coerced first via
    :func:`_coerce_number_text`; numbers (and ``None``) pass through
    unchanged, so this stays backward compatible for callers that already
    supply ``int`` / ``float``.
    """
    if isinstance(value, str):
        value = _coerce_number_text(value)
    if value is None:
        return None, f"{label} is required."
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{label} must be a number (received {value!r})."
    number = float(value)
    if not isfinite(number):
        return None, f"{label} must be a finite number (received {number})."
    if number < low:
        return None, (
            f"{label} value {number} must be {low} or greater "
            f"({unit})."
        )
    if high is not None and number > high:
        return None, (
            f"{label} value {number} is outside the allowed range "
            f"{low} to {high} {unit}."
        )
    return number, None


def parse_manual_entry(
    *,
    claimed_state: Any,
    observed: Any,
    expected: Any,
    tolerances: Any,
) -> tuple[ManualEntry | None, list[str]]:
    """Validate a manual submission.

    Returns ``(entry, errors)``. When ``errors`` is non-empty, ``entry`` is
    None and the caller must NOT run a verification - invalid input is never
    substituted with normal values.
    """
    errors: list[str] = []

    if claimed_state not in CLAIM_OPTIONS:
        errors.append(
            f"Controller-reported state must be one of: "
            f"{', '.join(CLAIM_OPTIONS)} (received {claimed_state!r})."
        )

    observed = observed if isinstance(observed, Mapping) else {}
    expected = expected if isinstance(expected, Mapping) else {}
    tolerances = tolerances if isinstance(tolerances, Mapping) else {}

    validated: dict[str, dict[str, float]] = {
        "observed": {},
        "expected": {},
        "tolerances": {},
    }

    for key in MEASUREMENT_KEYS:
        label = LABELS[key]
        unit = UNITS[key]
        low, high = PLAUSIBLE_RANGES[key]

        value, error = _check_number(
            observed.get(key),
            label=f"Observed {label}",
            unit=unit,
            low=low,
            high=high,
        )
        if error:
            errors.append(error)
        else:
            validated["observed"][key] = value

        value, error = _check_number(
            expected.get(key),
            label=f"Expected {label}",
            unit=unit,
            low=low,
            high=high,
        )
        if error:
            errors.append(error)
        else:
            validated["expected"][key] = value

        # Tolerances: non-negative and unbounded above.
        value, error = _check_number(
            tolerances.get(key),
            label=f"{label} tolerance",
            unit=unit,
            low=0.0,
            high=None,
        )
        if error:
            errors.append(error)
        else:
            validated["tolerances"][key] = value

    if errors:
        return None, errors

    return (
        ManualEntry(
            claimed_state=str(claimed_state),
            observed=validated["observed"],
            expected=validated["expected"],
            tolerances=validated["tolerances"],
        ),
        [],
    )
