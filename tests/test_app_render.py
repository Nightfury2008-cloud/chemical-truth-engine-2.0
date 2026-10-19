"""End-to-end render checks using Streamlit's AppTest.

These run app.py as a real Streamlit script (no browser needed) and catch
runtime errors in the presentation layer that unit tests of the engine cannot
see: exceptions while rendering, bad widget keys and session-state wiring.

They are intentionally small and fast; the heavy behavioural coverage lives in
the engine, simulation, CSV and serial tests.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from serial_source import pyserial_available
from simulation import SCENARIOS

# Absolute path so AppTest resolves app.py against the project root regardless
# of where pytest is invoked from.
APP = str(Path(__file__).resolve().parents[1] / "app.py")

# Sidebar navigation labels (the app prefixes each with an emoji).
NAV_LABELS = {
    "overview": "Overview",
    "demo": "Demonstration",
    "config": "Configuration",
    "history": "History & Charts",
    "methodology": "Methodology & Limitations",
}

SIM_MODE = "SIMULATION \U0001f9ea \u2014 synthetic software demo"
MANUAL_MODE = "MANUAL INPUT \u270d\ufe0f \u2014 type your own readings"
CSV_MODE = "CSV REPLAY \U0001f4c4 \u2014 recorded measurements file"
SERIAL_MODE = "ESP32 USB SERIAL \U0001f5a5\ufe0f \u2014 optional live hardware"


def _nav(app: AppTest, key: str) -> None:
    """Switch to a page via the sidebar navigation radio."""
    label = NAV_LABELS[key]
    nav = next(r for r in app.radio if r.label == "Navigation")
    option = next(o for o in nav.options if o.endswith(label))
    nav.set_value(option).run()
    assert not app.exception, f"Exception navigating to {key}"


def _page_text(app: AppTest) -> str:
    """Collect rendered text into one searchable string."""
    parts = []
    for block in (
        app.markdown,
        app.subheader,
        app.title,
        app.caption,
        app.warning,
        app.info,
        app.error,
        app.success,
        app.metric,
        app.code,
        app.text,
    ):
        for element in block:
            parts.append(str(element.value))
    return "\n".join(parts)


def _status_banner(app: AppTest) -> str:
    """Return the text of the status banner card (empty string if absent)."""
    for element in app.markdown:
        if "Current verification status" in str(element.value):
            return str(element.value)
    return ""


def _click_button(app: AppTest, needle: str) -> None:
    """Click a button whose label contains ``needle`` and re-run the script."""
    buttons = [b for b in app.button if needle in b.label]
    assert buttons, f"Button containing {needle!r} not found"
    buttons[0].click().run()
    assert not app.exception, f"Exception after clicking {needle!r}"


# --------------------------------------------------------------------------- #
# Smoke tests: the app must render every page without exceptions
# --------------------------------------------------------------------------- #

def test_app_renders_home_page_without_errors():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception, f"Runtime exception on load: {app.exception}"
    text = _page_text(app)
    assert "Chemical Truth Engine 2.0" in text
    assert "EDUCATIONAL SIMULATION" in text


def test_sidebar_lists_every_page():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    nav = next(r for r in app.radio if r.label == "Navigation")
    options = nav.options
    for label in NAV_LABELS.values():
        assert any(o.endswith(label) for o in options), (
            f"{label!r} missing from navigation: {options}"
        )


@pytest.mark.parametrize("key", list(NAV_LABELS))
def test_every_page_renders_without_exception(key):
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, key)
    assert not app.exception


# --------------------------------------------------------------------------- #
# Data-source mode selector
# --------------------------------------------------------------------------- #

def test_demo_page_offers_all_four_source_modes():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    source_radio = next(r for r in app.radio if r.label == "Data source")
    assert SIM_MODE in source_radio.options
    assert MANUAL_MODE in source_radio.options
    assert CSV_MODE in source_radio.options
    assert SERIAL_MODE in source_radio.options


def test_simulation_mode_shows_simulated_chip():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    assert "SIMULATED" in _page_text(app)


def test_csv_mode_renders_uploader_without_hardware():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    source_radio = next(r for r in app.radio if r.label == "Data source")
    source_radio.set_value(CSV_MODE).run()
    assert not app.exception
    assert app.file_uploader, "CSV mode must show a file uploader"
    text = _page_text(app).lower()
    assert "recorded" in text


def test_serial_mode_degrades_gracefully_without_pyserial():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    source_radio = next(r for r in app.radio if r.label == "Data source")
    source_radio.set_value(SERIAL_MODE).run()
    assert not app.exception
    text = _page_text(app)
    if pyserial_available():
        # pySerial present: the port-selection UI renders instead, still
        # with no automatic connection and no crash.
        assert "Read-only by design" in text
        assert "Connect" in text
    else:
        # pySerial absent: the app must explain that and keep working
        # rather than crash.
        assert "pySerial is not installed" in text
        assert "Simulation and CSV replay still work" in text
    # Always: the panel must state that a remotely hosted copy cannot reach
    # a USB device, so online visitors are not misled into expecting one.
    assert "cannot access a USB-connected ESP32" in text


# --------------------------------------------------------------------------- #
# Scenarios: each one must render the promised primary status
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "scenario_id, expected_status",
    [
        ("normal", "NORMAL"),
        ("sensor_mismatch", "INTEGRITY ANOMALY"),
        ("claim_mismatch", "INTEGRITY ANOMALY"),
        ("noisy", "NORMAL"),
        ("stale", "INSUFFICIENT DATA"),
        ("insufficient", "INSUFFICIENT DATA"),
    ],
)
def test_each_scenario_renders_its_promised_status(
    scenario_id, expected_status
):
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")

    scenario_select = app.selectbox[0]
    scenario_select.set_value(SCENARIOS[scenario_id]["name"]).run()
    assert not app.exception, f"Exception selecting {scenario_id}"

    _click_button(app, "Advance one sample")
    assert not app.exception, f"Exception running {scenario_id}"

    banner = _status_banner(app)
    assert banner, "Status banner not rendered after verification"
    assert expected_status in banner, (
        f"Expected {expected_status!r} in banner for {scenario_id}: {banner}"
    )


def test_multiple_advances_accumulate_history_and_trend():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    app.selectbox[0].set_value(SCENARIOS["normal"]["name"]).run()
    for _ in range(3):
        _click_button(app, "Advance one sample")
    assert not app.exception
    text = _page_text(app)
    assert "sample 3" in text.lower() or "sample #3" in text.lower()
    assert "Measurements across samples" in text


def test_anomaly_run_shows_table_and_attack_disclaimer():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    app.selectbox[0].set_value(SCENARIOS["sensor_mismatch"]["name"]).run()
    _click_button(app, "Advance one sample")
    text = _page_text(app)
    assert "not proof of a cyberattack" in text
    tables = [df.value for df in app.dataframe]
    comparison = next(t for t in tables if "Check" in t.columns)
    assert "Residual (observed - expected)" in comparison.columns
    assert "Tolerance (+/-)" in comparison.columns
    assert "Quality" in comparison.columns
    ph_row = comparison.set_index("Measurement").loc["pH"]
    assert ph_row["Check"] == "FAIL"
    other_rows = comparison.set_index("Measurement")
    assert other_rows.loc["Conductivity", "Check"] == "PASS"
    assert other_rows.loc["Temperature", "Check"] == "PASS"


def test_restart_simulation_keeps_history():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    app.selectbox[0].set_value(SCENARIOS["normal"]["name"]).run()
    _click_button(app, "Advance one sample")
    _click_button(app, "Advance one sample")
    _click_button(app, "Restart simulation")
    assert not app.exception
    assert "restarted" in _page_text(app).lower()
    # History survived the restart: run #2 is still recorded.
    _nav(app, "history")
    assert "2 run(s) recorded" in _page_text(app).lower()


def test_reruns_do_not_generate_new_samples():
    # Ordinary Streamlit reruns must not advance the simulation.
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    app.selectbox[0].set_value(SCENARIOS["normal"]["name"]).run()
    _click_button(app, "Advance one sample")
    before = _page_text(app)
    app.run()  # plain rerun
    assert not app.exception
    after = _page_text(app)
    assert before == after


# --------------------------------------------------------------------------- #
# Configuration page
# --------------------------------------------------------------------------- #

def test_configuration_page_renders_all_six_controls():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "config")
    assert not app.exception
    assert len(app.number_input) == 6


def test_configuration_page_survives_out_of_range_input():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "config")
    for widget in list(app.number_input):
        try:
            widget.set_value(-1.0).run()
        except Exception:
            pass
    assert not app.exception


def test_configuration_page_applies_new_value():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "config")
    app.number_input[0].set_value(7.0)  # expected pH
    _click_button(app, "Apply configuration")
    assert "applied" in _page_text(app).lower()


def test_restoring_defaults_is_possible():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "config")
    app.number_input[0].set_value(7.0)
    _click_button(app, "Apply configuration")
    _click_button(app, "Restore synthetic demo defaults")
    assert not app.exception


def test_configuration_page_shows_profiles_table():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "config")
    assert "Illustrative process profiles" in _page_text(app)
    frames = [df.value for df in app.dataframe]
    assert any("State" in f.columns for f in frames if hasattr(f, "columns"))


# --------------------------------------------------------------------------- #
# History page
# --------------------------------------------------------------------------- #

def test_history_page_renders_empty_state_without_error():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "history")
    assert not app.exception
    assert "empty" in _page_text(app).lower()


def test_history_page_shows_source_column_after_run():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    app.selectbox[0].set_value(SCENARIOS["normal"]["name"]).run()
    _click_button(app, "Advance one sample")
    _nav(app, "history")
    assert not app.exception
    text = _page_text(app).lower()
    assert "empty" not in text
    assert "run(s) recorded" in text
    history_table = next(
        (df.value for df in app.dataframe if "Source" in df.value.columns),
        None,
    )
    assert history_table is not None
    assert "SIMULATION" in history_table["Source"].tolist()


def test_history_page_two_runs_report_two_records():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    app.selectbox[0].set_value(SCENARIOS["normal"]["name"]).run()
    _click_button(app, "Advance one sample")
    _click_button(app, "Advance one sample")
    _nav(app, "history")
    assert not app.exception
    assert "2 run(s) recorded" in _page_text(app).lower()


def test_clear_history_flow_returns_to_empty_state():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    _nav(app, "demo")
    app.selectbox[0].set_value(SCENARIOS["normal"]["name"]).run()
    _click_button(app, "Advance one sample")
    _nav(app, "history")
    _click_button(app, "Clear demo history")
    assert not app.exception
    # Confirmation step appears before anything is deleted.
    assert "continue" in _page_text(app).lower()
    _click_button(app, "Yes, clear everything")
    assert not app.exception
    assert "empty" in _page_text(app).lower()
