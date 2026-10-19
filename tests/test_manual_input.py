"""Behavioural tests for MANUAL INPUT mode.

Part A tests manual_input.parse_manual_entry together with the authoritative
detection engine (no Streamlit needed). Part B drives the real dashboard
headlessly with Streamlit's AppTest: form submission, history recording and
result rendering.

The comparison logic itself lives in detection_engine.verify(); these tests
assert that manual input is validated, routed and reported correctly - never
that a second comparison algorithm exists.
"""

from math import inf, nan

import pytest

from streamlit.testing.v1 import AppTest

from detection_engine import (
    MEASUREMENT_KEYS,
    UNITS,
    VerificationStatus,
    verify,
)
from manual_input import CLAIM_OPTIONS, parse_manual_entry
from simulation import DEFAULT_EXPECTED, DEFAULT_TOLERANCES

# --------------------------------------------------------------------------- #
# Helpers (pure-Python part)
# --------------------------------------------------------------------------- #

CLAIM = "TARGET REACHED / NORMAL"


def _parse(**overrides):
    """parse_manual_entry with sensible defaults; keyword overrides."""
    args = {
        "claimed_state": CLAIM,
        "observed": dict(DEFAULT_EXPECTED),
        "expected": dict(DEFAULT_EXPECTED),
        "tolerances": dict(DEFAULT_TOLERANCES),
    }
    args.update(overrides)
    return parse_manual_entry(**args)


def _verify(entry):
    return verify(
        entry.observed,
        entry.expected,
        entry.tolerances,
        entry.claimed_state,
    )


def _check(result, key):
    return next(c for c in result.checks if c.key == key)


# --------------------------------------------------------------------------- #
# Part A - validation (manual_input.parse_manual_entry)
# --------------------------------------------------------------------------- #


def test_default_entry_parses_without_errors():
    entry, errors = _parse()
    assert errors == []
    assert entry is not None
    assert entry.claimed_state == CLAIM
    assert entry.observed == DEFAULT_EXPECTED
    assert entry.tolerances == DEFAULT_TOLERANCES


def test_invalid_entry_returns_none_entry():
    entry, errors = _parse(observed={**DEFAULT_EXPECTED, "ph": None})
    assert entry is None
    assert errors


def test_missing_observed_value_is_rejected():
    observed = dict(DEFAULT_EXPECTED)
    observed["ph"] = None
    entry, errors = _parse(observed=observed)
    assert entry is None
    assert any("pH is required" in e for e in errors)


def test_missing_key_entirely_is_rejected():
    entry, errors = _parse(observed={"ph": 8.0, "temperature": 25.5})
    assert entry is None
    assert any("Conductivity is required" in e for e in errors)


def test_missing_expected_and_tolerances_mappings_are_rejected():
    entry, errors = _parse(expected=None, tolerances=None)
    assert entry is None
    # Three expected + three tolerance errors, none silently defaulted.
    assert sum("is required" in e for e in errors) == 6


def test_nonnumeric_value_is_rejected():
    observed = dict(DEFAULT_EXPECTED)
    observed["ph"] = "acid"
    entry, errors = _parse(observed=observed)
    assert entry is None
    assert any("must be a number" in e for e in errors)


@pytest.mark.parametrize("bad", [nan, inf, -inf])
def test_nan_and_infinite_values_are_rejected(bad):
    observed = dict(DEFAULT_EXPECTED)
    observed["conductivity"] = bad
    entry, errors = _parse(observed=observed)
    assert entry is None
    assert any("finite" in e for e in errors)


def test_boolean_value_is_not_accepted_as_a_number():
    tolerances = dict(DEFAULT_TOLERANCES)
    tolerances["ph"] = True  # bool must not sneak through as 1.0
    entry, errors = _parse(tolerances=tolerances)
    assert entry is None
    assert any("must be a number" in e for e in errors)


@pytest.mark.parametrize("bad", [14.1, -0.1])
def test_ph_outside_0_to_14_is_rejected(bad):
    observed = dict(DEFAULT_EXPECTED)
    observed["ph"] = bad
    entry, errors = _parse(observed=observed)
    assert entry is None
    assert any("pH" in e for e in errors)


def test_negative_conductivity_is_rejected():
    observed = dict(DEFAULT_EXPECTED)
    observed["conductivity"] = -1.0
    entry, errors = _parse(observed=observed)
    assert entry is None
    assert any("Conductivity" in e and "greater" in e for e in errors)


@pytest.mark.parametrize("bad", [-60.0, 250.0])
def test_temperature_outside_sensor_limits_is_rejected(bad):
    observed = dict(DEFAULT_EXPECTED)
    observed["temperature"] = bad
    entry, errors = _parse(observed=observed)
    assert entry is None
    assert any("Temperature" in e for e in errors)


@pytest.mark.parametrize("bad", [-0.1, -5.0])
def test_negative_tolerance_is_rejected(bad):
    tolerances = dict(DEFAULT_TOLERANCES)
    tolerances["ph"] = bad
    entry, errors = _parse(tolerances=tolerances)
    assert entry is None
    assert any("tolerance" in e and "greater" in e for e in errors)


def test_unknown_claim_string_is_rejected():
    entry, errors = _parse(claimed_state="everything is fine")
    assert entry is None
    assert any("Controller-reported state" in e for e in errors)


def test_zero_tolerance_is_allowed():
    tolerances = {k: 0.0 for k in MEASUREMENT_KEYS}
    entry, errors = _parse(tolerances=tolerances)
    assert errors == []
    assert entry.tolerances == tolerances


# --------------------------------------------------------------------------- #
# Part A - verification semantics through the engine
# --------------------------------------------------------------------------- #


def test_all_within_tolerance_is_normal():
    observed = {"ph": 8.3, "conductivity": 700.0, "temperature": 26.9}
    entry, errors = _parse(observed=observed)
    assert errors == []
    result = _verify(entry)
    assert result.status is VerificationStatus.NORMAL
    assert all(c.passed for c in result.checks)


def test_out_of_tolerance_ph_is_anomaly():
    observed = dict(DEFAULT_EXPECTED)
    observed["ph"] = 9.5  # residual +1.5 vs tolerance 0.5
    entry, _ = _parse(observed=observed)
    result = _verify(entry)
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY
    assert _check(result, "ph").passed is False
    assert _check(result, "conductivity").passed is True
    assert _check(result, "temperature").passed is True


def test_out_of_tolerance_conductivity_is_anomaly():
    observed = dict(DEFAULT_EXPECTED)
    observed["conductivity"] = 800.0  # residual +150 vs tolerance 80
    entry, _ = _parse(observed=observed)
    result = _verify(entry)
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY
    assert _check(result, "conductivity").passed is False
    assert _check(result, "ph").passed is True


def test_out_of_tolerance_temperature_is_anomaly():
    observed = dict(DEFAULT_EXPECTED)
    observed["temperature"] = 30.0  # residual +4.5 vs tolerance 2.0
    entry, _ = _parse(observed=observed)
    result = _verify(entry)
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY
    assert _check(result, "temperature").passed is False
    assert _check(result, "ph").passed is True


def test_residual_exactly_at_tolerance_is_accepted():
    # Inclusive rule: |residual| <= tolerance passes, even exactly equal.
    observed = {"ph": 8.5, "conductivity": 730.0, "temperature": 27.5}
    entry, _ = _parse(observed=observed)
    result = _verify(entry)
    assert result.status is VerificationStatus.NORMAL
    assert _check(result, "ph").residual == pytest.approx(0.5)
    assert _check(result, "conductivity").residual == pytest.approx(80.0)
    assert _check(result, "temperature").residual == pytest.approx(2.0)


def test_just_beyond_tolerance_fails():
    observed = dict(DEFAULT_EXPECTED)
    observed["ph"] = 8.5001
    entry, _ = _parse(observed=observed)
    assert _verify(entry).status is VerificationStatus.INTEGRITY_ANOMALY


def test_zero_tolerance_with_exact_match_is_normal():
    tolerances = {k: 0.0 for k in MEASUREMENT_KEYS}
    entry, _ = _parse(tolerances=tolerances)
    assert _verify(entry).status is VerificationStatus.NORMAL


def test_zero_tolerance_with_tiny_deviation_is_anomaly():
    tolerances = {k: 0.0 for k in MEASUREMENT_KEYS}
    observed = dict(DEFAULT_EXPECTED)
    observed["ph"] = 8.001
    entry, _ = _parse(observed=observed, tolerances=tolerances)
    assert _verify(entry).status is VerificationStatus.INTEGRITY_ANOMALY


def test_custom_wider_tolerance_turns_anomaly_into_normal():
    observed = dict(DEFAULT_EXPECTED)
    observed["ph"] = 9.5  # +1.5: fails 0.5 tolerance, passes 2.0 tolerance
    strict, _ = _parse(observed=observed)
    assert _verify(strict).status is VerificationStatus.INTEGRITY_ANOMALY
    tolerant, _ = _parse(
        observed=observed,
        tolerances={**DEFAULT_TOLERANCES, "ph": 2.0},
    )
    assert _verify(tolerant).status is VerificationStatus.NORMAL


def test_missing_value_can_never_produce_normal():
    # Defence in depth: even if invalid data bypassed the form validation,
    # the engine must answer INSUFFICIENT DATA, never NORMAL.
    observed = {"ph": None, "conductivity": 650.0, "temperature": 25.5}
    result = verify(observed, DEFAULT_EXPECTED, DEFAULT_TOLERANCES, CLAIM)
    assert result.status is VerificationStatus.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# Part A - controller-claim semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("claim", CLAIM_OPTIONS)
def test_claim_alone_never_produces_anomaly(claim):
    # Matching numbers stay NORMAL for every claim - a non-normal claim is
    # never automatically treated as an attack or as inconsistent.
    entry, errors = _parse(claimed_state=claim)
    assert errors == []
    assert _verify(entry).status is VerificationStatus.NORMAL


@pytest.mark.parametrize("claim", CLAIM_OPTIONS)
def test_every_claim_is_recorded_verbatim(claim):
    entry, _ = _parse(claimed_state=claim)
    assert _verify(entry).claimed_state == claim


def test_target_claim_with_violating_readings_is_anomaly():
    # Controller claims the target was reached, but the numbers say
    # otherwise: the engine flags the contradiction.
    observed = {"ph": 7.0, "conductivity": 450.0, "temperature": 24.0}
    entry, _ = _parse(claimed_state="TARGET REACHED / NORMAL",
                      observed=observed)
    result = _verify(entry)
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY


# --------------------------------------------------------------------------- #
# Part A - residuals and units
# --------------------------------------------------------------------------- #


def test_residuals_are_observed_minus_expected_with_correct_units():
    observed = {"ph": 9.0, "conductivity": 700.0, "temperature": 27.5}
    expected = {"ph": 8.0, "conductivity": 650.0, "temperature": 25.5}
    entry, _ = _parse(observed=observed, expected=expected)
    result = _verify(entry)
    by_key = {c.key: c for c in result.checks}
    assert by_key["ph"].residual == pytest.approx(1.0)
    assert by_key["conductivity"].residual == pytest.approx(50.0)
    assert by_key["temperature"].residual == pytest.approx(2.0)
    # Units are the project-standard ones for each measurement.
    assert UNITS == {
        "ph": "dimensionless",
        "conductivity": "\u00b5S/cm",
        "temperature": "\u00b0C",
    }
    for key in MEASUREMENT_KEYS:
        assert by_key[key].unit == UNITS[key]


# --------------------------------------------------------------------------- #
# Part B - dashboard behaviour (AppTest, no browser)
# --------------------------------------------------------------------------- #

from pathlib import Path  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
MANUAL_MODE = "MANUAL INPUT \u270d\ufe0f \u2014 type your own readings"
SIM_MODE = "SIMULATION \U0001f9ea \u2014 synthetic software demo"


def _manual_app() -> AppTest:
    """AppTest sitting on the Demonstration page in MANUAL INPUT mode."""
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    nav = next(r for r in app.radio if r.label == "Navigation")
    nav.set_value(
        next(o for o in nav.options if o.endswith("Demonstration"))
    ).run()
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(MANUAL_MODE).run()
    assert not app.exception
    return app


#: Reference-source option labels (mirrors app.py; kept literal so a
#: silent change to the app's wording fails these tests loudly).
REF_TYPE_MYSELF = "Enter expected values and tolerances myself"
REF_CONFIG = "Configuration page values (expected + tolerances)"
REF_STEADY = (
    "Illustrative profile: Steady operation "
    "(expected; tolerances from Configuration page)"
)
REF_TARGET = (
    "Illustrative profile: Target state reached "
    "(expected; tolerances from Configuration page)"
)

# Provenance labels recorded with each run (short forms shown in results).
PROV_MYSELF = "typed into the form"
PROV_CONFIG = "Configuration page values"
PROV_STEADY = (
    "illustrative profile \u2018Steady operation\u2019 (expected) + "
    "Configuration page (tolerances)"
)
PROV_TARGET = (
    "illustrative profile \u2018Target state reached\u2019 (expected) + "
    "Configuration page (tolerances)"
)


def _set_reference(app: AppTest, option: str) -> None:
    """Pick the reference-source selector; its callback fills/clears fields."""
    selector = next(
        s
        for s in app.selectbox
        if s.label == "Expected values and tolerances come from:"
    )
    selector.set_value(option).run()
    assert not app.exception


def _fill_observed(
    app: AppTest,
    ph: float | None = 8.2,
    cond: float | None = 650.0,
    temp: float | None = 25.5,
) -> None:
    """Set the three observed fields (None clears a field back to empty)."""
    for key, value in (
        ("man_obs_ph", ph),
        ("man_obs_cond", cond),
        ("man_obs_temp", temp),
    ):
        if value is None:
            continue
        _set_input(app, key, value)


def _widget(app: AppTest, key: str):
    return next(
        (w for w in app.text_input if getattr(w, "key", None) == key), None
    )


def _set_input(app: AppTest, key: str, value: float | str) -> None:
    widget = _widget(app, key)
    assert widget is not None, f"text_input {key!r} not found"
    # Manual fields are text inputs; send the value as the text a user
    # would type. Numbers are stringified (e.g. 8.2 -> "8.2").
    widget.set_value(str(value) if not isinstance(value, str) else value)


def _num(app: AppTest, key: str) -> float | None:
    """Widget content as a number; None when the field is blank."""
    widget = _widget(app, key)
    assert widget is not None, f"text_input {key!r} not found"
    text = widget.value
    if text is None or str(text).strip() == "":
        return None
    return float(text)


def _is_blank(app: AppTest, key: str) -> bool:
    return _num(app, key) is None


def _submit(app: AppTest) -> None:
    button = next(b for b in app.button if b.label == "Verify Measurements")
    button.click().run()
    assert not app.exception


def _status_banner(app: AppTest) -> str:
    for element in app.markdown:
        if "Current verification status" in str(element.value):
            return str(element.value)
    return ""


def test_manual_mode_shows_the_required_notice():
    app = _manual_app()
    texts = [str(i.value) for i in app.info]
    assert any(
        "MANUAL ENTRY MODE" in t
        and "not automatically verified physical measurements" in t
        for t in texts
    )


def test_manual_mode_offers_the_four_claims():
    app = _manual_app()
    claim_box = next(
        s for s in app.selectbox if s.label == "Controller-reported state"
    )
    assert list(claim_box.options) == list(CLAIM_OPTIONS)


def test_manual_submission_adds_a_history_record():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.2)
    _submit(app)
    history = app.session_state["history"]
    assert len(history) == 1
    record = history[0]
    assert record["source"] == "MANUAL INPUT"
    assert record["claimed_state"] == "TARGET REACHED / NORMAL"
    assert record["result"] == "NORMAL"
    assert record["observed"]["ph"] == pytest.approx(8.2)
    assert record["residuals"]["ph"] == pytest.approx(0.2)
    # The reference source the user explicitly picked is recorded.
    assert record["reference_source"] == PROV_CONFIG


def test_manual_submission_renders_residuals_and_units_in_table():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=9.5)
    _submit(app)
    frames = [df.value for df in app.dataframe]
    table = next(f for f in frames if "Check" in f.columns)
    row = table.set_index("Measurement").loc["pH"]
    assert row["Unit"] == "dimensionless"
    assert row["Expected"] == "8.00"
    assert row["Observed"] == "9.50"
    assert row["Residual (observed - expected)"] == "1.50"
    assert row["Tolerance (+/-)"] == "0.50"
    assert row["Check"] == "FAIL"


def test_manual_anomaly_submission_shows_status_and_disclaimer():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=9.5)
    _submit(app)
    assert "INTEGRITY ANOMALY" in _status_banner(app)
    # The shared result view carries the standing anomaly reminder...
    warning_texts = " ".join(str(w.value) for w in app.warning)
    assert "not proof of a cyberattack" in warning_texts
    # ...and the manual panel adds the required note verbatim.
    caption_texts = " ".join(str(c.value) for c in app.caption)
    assert (
        "An integrity anomaly indicates a mismatch with the configured "
        "expected response. It does not prove that a cyberattack occurred."
    ) in caption_texts


def test_manual_custom_tolerance_submission():
    # Same deviation as the anomaly test, but the user widened the pH
    # tolerance to 2.0: residual 1.5 now passes.
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=9.5)
    _set_input(app, "man_tol_ph", 2.0)
    _submit(app)
    assert "NORMAL" in _status_banner(app)
    record = app.session_state["history"][0]
    assert record["tolerances"]["ph"] == pytest.approx(2.0)
    # The submitted tolerance no longer matches the loaded snapshot, so the
    # provenance is honest about the edit.
    assert "edited in the form" in record["reference_source"]


def test_repeated_manual_submissions_preserve_earlier_history():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.1)
    _submit(app)
    _fill_observed(app, ph=9.5)
    _submit(app)
    history = app.session_state["history"]
    assert len(history) == 2
    # The first record is untouched by the second submission.
    assert history[0]["run"] == 1
    assert history[0]["observed"]["ph"] == pytest.approx(8.1)
    assert history[0]["result"] == "NORMAL"
    assert history[1]["run"] == 2
    assert history[1]["observed"]["ph"] == pytest.approx(9.5)
    assert history[1]["result"] == "INTEGRITY ANOMALY"


def test_manual_runs_appear_in_the_history_table_with_source():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app)
    _submit(app)
    nav = next(r for r in app.radio if r.label == "Navigation")
    nav.set_value(
        next(o for o in nav.options if o.endswith("History & Charts"))
    ).run()
    assert not app.exception
    table = next(df.value for df in app.dataframe if "Source" in df.value.columns)
    assert table["Source"].tolist() == ["MANUAL INPUT"]
    assert table["Reference"].tolist() == [PROV_CONFIG]


def test_switching_to_simulation_after_manual_still_works():
    # Old callers of the shared verification funnel keep working: a manual
    # run followed by a simulation run both land in history correctly.
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app)
    _submit(app)
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(SIM_MODE).run()
    advance = next(b for b in app.button if "Advance one sample" in b.label)
    advance.click().run()
    assert not app.exception
    history = app.session_state["history"]
    assert [r["source"] for r in history] == ["MANUAL INPUT", "SIMULATION"]


def test_plain_rerun_does_not_add_manual_records():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app)
    _submit(app)
    before = len(app.session_state["history"])
    app.run()
    assert not app.exception
    assert len(app.session_state["history"]) == before


# --------------------------------------------------------------------------- #
# Part C - user-data-driven manual mode
# --------------------------------------------------------------------------- #

MANUAL_KEYS = (
    "man_obs_ph",
    "man_obs_cond",
    "man_obs_temp",
    "man_exp_ph",
    "man_exp_cond",
    "man_exp_temp",
    "man_tol_ph",
    "man_tol_cond",
    "man_tol_temp",
)


def test_manual_fields_open_empty():
    # Every field starts blank - no demo values are smuggled in.
    app = _manual_app()
    for key in MANUAL_KEYS:
        widget = _widget(app, key)
        assert widget is not None, f"text_input {key!r} not found"
        assert widget.value in (None, ""), (
            f"{key} should open empty, got {widget.value!r}"
        )


def test_manual_fields_are_editable_text_inputs_not_empty_number_inputs():
    # ROOT-CAUSE REGRESSION: the nine manual fields must be plain text
    # inputs. They were previously st.number_input(value=None) "clearable"
    # widgets, which do not reliably accept typed input in the browser
    # while st.number_input with a real default value works fine (the
    # Configuration page). Pinning the widget type prevents a silent
    # revert to the broken configuration.
    app = _manual_app()
    for key in MANUAL_KEYS:
        assert _widget(app, key) is not None, (
            f"{key} missing from app.text_input"
        )
        assert not any(
            getattr(w, "key", None) == key for w in app.number_input
        ), f"{key} must not be a number_input"


def test_manual_fields_are_enabled_and_not_read_only():
    # The widgets must be enabled; a disabled or read-only field would
    # reproduce the "cannot enter data" report even though it renders.
    app = _manual_app()
    for key in MANUAL_KEYS:
        widget = _widget(app, key)
        assert getattr(widget, "disabled", False) is False, (
            f"{key} is disabled"
        )


def test_manual_text_fields_accept_decimals_and_submit_uses_them():
    # Typed decimal text must reach the engine as the exact value entered
    # (residual = observed - expected), not be dropped or reformatted.
    app = _manual_app()
    _set_reference(app, REF_CONFIG)  # fills expected/tolerances as strings
    _set_input(app, "man_obs_ph", "8.20")
    _set_input(app, "man_obs_cond", "650.0")
    _set_input(app, "man_obs_temp", "25.5")
    _submit(app)
    assert not app.exception
    history = app.session_state["history"]
    assert len(history) == 1
    record = history[0]
    assert record["observed"]["ph"] == pytest.approx(8.20)
    assert record["observed"]["conductivity"] == pytest.approx(650.0)
    assert record["observed"]["temperature"] == pytest.approx(25.5)
    assert record["residuals"]["ph"] == pytest.approx(0.20)
    assert record["result"] == "NORMAL"


def test_manual_text_field_non_numeric_is_rejected():
    # Garbage text must be reported, never coerced into a normal number.
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _set_input(app, "man_obs_ph", "acid")
    _submit(app)
    assert app.session_state["history"] == []
    error_text = " ".join(str(e.value) for e in app.error)
    assert "Observed pH" in error_text
    assert "must be a number" in error_text


def test_manual_text_field_whitespace_only_is_required():
    # Whitespace-only input counts as empty, not as a valid zero.
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _set_input(app, "man_obs_ph", "   ")
    _submit(app)
    assert app.session_state["history"] == []
    error_text = " ".join(str(e.value) for e in app.error)
    assert "Observed pH is required" in error_text


def test_manual_text_field_negative_conductivity_is_rejected():
    # Range validation must still apply to typed text (no min/max on the
    # widget now, so the validator is the enforcement point).
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.2, cond=-5.0, temp=25.5)
    _submit(app)
    assert app.session_state["history"] == []
    error_text = " ".join(str(e.value) for e in app.error)
    assert "Conductivity" in error_text
    assert "greater" in error_text


def test_opening_manual_mode_records_nothing_and_never_verifies():
    # Merely navigating into manual mode (and plain reruns) must not create
    # history, results or chart points.
    app = _manual_app()
    app.run()
    assert not app.exception
    assert app.session_state["history"] == []
    assert app.session_state["chart_points"] == []
    assert "last_runs" not in app.session_state or all(
        v is None for v in app.session_state["last_runs"].values()
    )
    texts = " ".join(str(i.value) for i in app.info)
    assert "No manual verification yet" in texts
    assert _status_banner(app) == ""


def test_empty_submit_is_rejected_and_records_nothing():
    # Requirement: empty manual inputs can never produce NORMAL.
    app = _manual_app()
    _submit(app)
    assert app.session_state["history"] == []
    error_text = " ".join(str(e.value) for e in app.error)
    for field in (
        "Observed pH", "Observed Conductivity", "Observed Temperature",
        "Expected pH", "Expected Conductivity", "Expected Temperature",
        "pH tolerance", "Conductivity tolerance", "Temperature tolerance",
    ):
        assert field in error_text, f"missing error for {field}"
    assert "NORMAL" not in _status_banner(app)
    texts = " ".join(str(w.value) for w in app.warning)
    assert "No verification was recorded" in texts


def test_observed_only_submit_is_rejected_missing_references_not_substituted():
    # Observed values alone must NOT be silently verified against demo or
    # Configuration values - the missing references are reported instead.
    app = _manual_app()
    _fill_observed(app, ph=8.2)
    _submit(app)
    assert app.session_state["history"] == []
    error_text = " ".join(str(e.value) for e in app.error)
    assert "Expected pH" in error_text
    assert "Expected Conductivity" in error_text
    assert "Expected Temperature" in error_text
    assert _status_banner(app) == ""


def test_reference_selector_configuration_fills_only_reference_fields():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    for key, expected_value in (
        ("man_exp_ph", 8.0), ("man_exp_cond", 650.0), ("man_exp_temp", 25.5),
        ("man_tol_ph", 0.5), ("man_tol_cond", 80.0), ("man_tol_temp", 2.0),
    ):
        assert _num(app, key) == pytest.approx(expected_value)
    # Observed readings are NEVER touched by the reference loader.
    for key in ("man_obs_ph", "man_obs_cond", "man_obs_temp"):
        assert _is_blank(app, key), f"{key} was overwritten"


def test_reference_selector_profile_fills_profile_expected_values():
    app = _manual_app()
    _set_reference(app, REF_TARGET)
    for key, expected_value in (
        ("man_exp_ph", 7.2), ("man_exp_cond", 520.0), ("man_exp_temp", 26.2),
    ):
        assert _num(app, key) == pytest.approx(expected_value)
    # Tolerances still come from the Configuration page (stated in the label).
    assert _num(app, "man_tol_ph") == pytest.approx(0.5)
    assert _is_blank(app, "man_obs_ph")
    # The steady profile offers its own expected values too.
    _set_reference(app, REF_STEADY)
    assert _num(app, "man_exp_ph") == pytest.approx(8.0)
    assert _num(app, "man_exp_cond") == pytest.approx(650.0)
    assert _num(app, "man_exp_temp") == pytest.approx(25.5)


def test_reference_selector_myself_clears_reference_fields():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _set_reference(app, REF_TYPE_MYSELF)
    for key in ("man_exp_ph", "man_tol_ph", "man_tol_cond", "man_tol_temp"):
        assert _is_blank(app, key), f"{key} should be cleared"


def test_profile_reference_submission_records_exact_provenance():
    # A run verified against the steady profile records exactly where its
    # reference values came from.
    app = _manual_app()
    _set_reference(app, REF_STEADY)
    _fill_observed(app, ph=8.1, cond=645.0, temp=25.4)
    _submit(app)
    record = app.session_state["history"][0]
    assert record["result"] == "NORMAL"
    assert record["reference_source"] == PROV_STEADY
    assert record["expected"]["ph"] == pytest.approx(8.0)


def test_typed_expected_values_drive_the_residuals():
    # Residuals come from the submitted numbers, not from any preset.
    app = _manual_app()
    _fill_observed(app, ph=8.2)
    _set_input(app, "man_exp_ph", 7.0)
    _set_input(app, "man_tol_ph", 0.3)
    _set_input(app, "man_exp_cond", 650.0)
    _set_input(app, "man_tol_cond", 80.0)
    _set_input(app, "man_exp_temp", 25.5)
    _set_input(app, "man_tol_temp", 2.0)
    _submit(app)
    record = app.session_state["history"][0]
    assert record["expected"]["ph"] == pytest.approx(7.0)
    assert record["residuals"]["ph"] == pytest.approx(1.2)
    assert record["result"] == "INTEGRITY ANOMALY"
    assert record["reference_source"] == PROV_MYSELF


def test_changed_tolerance_flips_the_result():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.9)
    _set_input(app, "man_tol_ph", 0.5)
    _submit(app)
    assert app.session_state["history"][0]["result"] == "INTEGRITY ANOMALY"
    _set_input(app, "man_tol_ph", 1.0)
    _submit(app)
    assert app.session_state["history"][1]["result"] == "NORMAL"


def test_changed_value_flips_the_result():
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.1)
    _submit(app)
    assert app.session_state["history"][0]["result"] == "NORMAL"
    _fill_observed(app, ph=9.9)
    _submit(app)
    assert app.session_state["history"][1]["result"] == "INTEGRITY ANOMALY"


def test_target_claim_with_conflicting_profile_readings_is_anomaly():
    # The controller claims the target state; the user loads that state's
    # illustrative profile as the reference and the readings disagree.
    app = _manual_app()
    _set_reference(app, REF_TARGET)
    _fill_observed(app, ph=8.2, cond=650.0, temp=25.5)
    _submit(app)
    record = app.session_state["history"][0]
    assert record["claimed_state"] == "TARGET REACHED / NORMAL"
    assert record["result"] == "INTEGRITY ANOMALY"
    assert record["reference_source"] == PROV_TARGET


def test_non_normal_claim_with_matching_numbers_is_normal():
    # A non-normal controller claim is never auto-flagged as an attack:
    # only the numbers decide the status.
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.1, cond=640.0, temp=25.4)
    # The claim lives inside the form; set it just before submitting so the
    # form's state commit includes it.
    claim_box = next(
        s for s in app.selectbox if s.label == "Controller-reported state"
    )
    claim_box.set_value("FAULT REPORTED")
    _submit(app)
    record = app.session_state["history"][0]
    assert record["claimed_state"] == "FAULT REPORTED"
    assert record["result"] == "NORMAL"


def test_unknown_claim_with_violating_numbers_is_anomaly_not_attack():
    # UNKNOWN claim + conflicting numbers -> INTEGRITY ANOMALY (numbers
    # only), never anything mentioning an attacker.
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=12.0)
    claim_box = next(
        s for s in app.selectbox if s.label == "Controller-reported state"
    )
    claim_box.set_value("UNKNOWN")
    _submit(app)
    record = app.session_state["history"][0]
    assert record["claimed_state"] == "UNKNOWN"
    assert record["result"] == "INTEGRITY ANOMALY"
    page_text = " ".join(str(m.value) for m in app.markdown)
    assert "attacker" not in page_text.lower()


def test_manual_result_is_isolated_from_simulation_result():
    # Per-source results: a simulation run never appears while the manual
    # form is shown, and vice versa.
    app = _manual_app()
    # Produce a simulation run first.
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(SIM_MODE).run()
    advance = next(b for b in app.button if "Advance one sample" in b.label)
    advance.click().run()
    assert not app.exception
    assert app.session_state["history"]
    # Switch to manual: the result area shows an empty state, not the
    # simulation's metric cards.
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(MANUAL_MODE).run()
    texts = " ".join(str(i.value) for i in app.info)
    assert "No manual verification yet" in texts
    assert _status_banner(app) == ""
    # After a manual run, the manual view shows the manual record.
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.2)
    _submit(app)
    assert "Run #2" in " ".join(str(s.value) for s in app.subheader)
    # Back to simulation: its own latest record is shown again (run #1).
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(SIM_MODE).run()
    assert "source: SIMULATION" in " ".join(
        str(s.value) for s in app.subheader
    )


def test_manual_chart_points_carry_only_manual_values():
    app = _manual_app()
    # One simulation run, then two manual runs.
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(SIM_MODE).run()
    advance = next(b for b in app.button if "Advance one sample" in b.label)
    advance.click().run()
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(MANUAL_MODE).run()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.1)
    _submit(app)
    _fill_observed(app, ph=9.9)
    _submit(app)
    points = app.session_state["chart_points"]
    manual_points = [p for p in points if p.get("source") == "MANUAL INPUT"]
    sim_points = [p for p in points if p.get("source") == "SIMULATION"]
    assert len(sim_points) == 1
    assert len(manual_points) == 2
    # The manual chart points hold exactly the user's typed values.
    assert [p["observed"]["ph"] for p in manual_points] == pytest.approx(
        [8.1, 9.9]
    )


def test_invalid_submission_clears_the_displayed_manual_result():
    # After a successful manual run, an invalid submit must not leave the
    # previous result on screen as if it had passed.
    app = _manual_app()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.1)
    _submit(app)
    assert app.session_state["history"]
    # Now clear an observed field and submit again.
    _set_input(app, "man_obs_ph", 8.1)
    widget = _widget(app, "man_obs_cond")
    widget.set_value(None)  # type: ignore[arg-type]
    _submit(app)
    assert len(app.session_state["history"]) == 1  # nothing new recorded
    texts = " ".join(str(i.value) for i in app.info)
    assert "No manual verification yet" in texts
    assert _status_banner(app) == ""


def test_history_source_filter_plots_only_selected_source():
    # The History page chart filter plots only the runs of the chosen
    # source, while the table lists everything.
    app = _manual_app()
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(SIM_MODE).run()
    advance = next(b for b in app.button if "Advance one sample" in b.label)
    advance.click().run()
    source = next(r for r in app.radio if r.label == "Data source")
    source.set_value(MANUAL_MODE).run()
    _set_reference(app, REF_CONFIG)
    _fill_observed(app, ph=8.2)
    _submit(app)
    nav = next(r for r in app.radio if r.label == "Navigation")
    nav.set_value(
        next(o for o in nav.options if o.endswith("History & Charts"))
    ).run()
    assert not app.exception
    selector = next(
        s for s in app.selectbox if s.label == "Trend chart data source"
    )
    assert list(selector.options) == ["All sources", "SIMULATION", "MANUAL INPUT"]
    selector.set_value("MANUAL INPUT").run()
    assert not app.exception
    captions = " ".join(str(c.value) for c in app.caption)
    assert "1 of 2 run(s)" in captions
    # The history table still lists both runs.
    table = next(df.value for df in app.dataframe if "Source" in df.value.columns)
    assert table["Source"].tolist() == ["SIMULATION", "MANUAL INPUT"]
