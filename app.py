"""Chemical Truth Engine 2.0 - Streamlit dashboard (presentation layer).

This file only handles layout, user interaction and session state. All
comparison logic lives in detection_engine.py, all synthetic generation in
simulation.py, CSV validation in csv_replay.py and the optional read-only
serial integration in serial_source.py - so every part can be unit-tested
without launching Streamlit.

Data-source modes:
* SIMULATION    - synthetic software demo, works with no hardware at all.
* MANUAL INPUT  - readings typed by the user, verified through the engine.
* CSV REPLAY    - replay a recorded measurements file.
* ESP32 SERIAL  - optional, read-only USB serial input (pySerial required).

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

try:  # Plotly is optional: the charts fall back to Streamlit's own charts.
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_PLOTLY = False

from csv_replay import REQUIRED_COLUMNS, parse_csv
from detection_engine import (
    DECIMALS,
    LABELS,
    MEASUREMENT_KEYS,
    UNITS,
    validate_configuration,
    verify,
)
from history import (
    append_record,
    build_record,
    clear_history,
    next_run_number,
    to_dataframe,
    trend_dataframe,
)
from manual_input import CLAIM_NOTES, CLAIM_OPTIONS, parse_manual_entry
from process_profiles import PROFILES
from serial_source import (
    DEFAULT_BAUDRATE,
    DEFAULT_STALENESS_SECONDS,
    SerialReader,
    is_stale,
    list_ports,
    pyserial_available,
)
from simulation import (
    DEFAULT_EXPECTED,
    DEFAULT_SEED,
    DEFAULT_TOLERANCES,
    SCENARIOS,
    make_scenario_simulator,
)

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Chemical Truth Engine 2.0",
    page_icon="\u2697\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: 0.2px; }
    div[data-testid="stMetric"] {
        background: #141a24;
        border: 1px solid #25303f;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #9fb0c8; }
    div[data-testid="stMetricValue"] { color: #f2f5fa; }
    .cte-banner {
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid;
        min-height: 92px;
    }
    .cte-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .cte-status { font-size: 1.3rem; font-weight: 700; }
    .cte-note {
        background: #10161f;
        border: 1px solid #25303f;
        border-left: 4px solid #4f8cff;
        border-radius: 8px;
        padding: 12px 16px;
        color: #d6dde8;
    }
    .cte-diagram {
        background: #0d1218;
        border: 1px solid #25303f;
        border-radius: 10px;
        padding: 16px 18px;
        color: #cfd8e6;
        font-size: 0.85rem;
    }
    .cte-chip {
        display: inline-block;
        border-radius: 6px;
        padding: 2px 8px;
        margin-right: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session-state initialisation (temporary demo memory - never persisted)
# --------------------------------------------------------------------------- #

_CFG_KEYS = {
    ("expected", "ph"): "cfg_exp_ph",
    ("expected", "conductivity"): "cfg_exp_conductivity",
    ("expected", "temperature"): "cfg_exp_temperature",
    ("tolerances", "ph"): "cfg_tol_ph",
    ("tolerances", "conductivity"): "cfg_tol_conductivity",
    ("tolerances", "temperature"): "cfg_tol_temperature",
}

#: Data-source modes offered by the dashboard.
SOURCE_MODES = {
    "SIMULATION \U0001f9ea \u2014 synthetic software demo": "simulation",
    "MANUAL INPUT \u270d\ufe0f \u2014 type your own readings": "manual",
    "CSV REPLAY \U0001f4c4 \u2014 recorded measurements file": "csv",
    "ESP32 USB SERIAL \U0001f5a5\ufe0f \u2014 optional live hardware": "serial",
}

SOURCE_LABELS = {
    "simulation": "SIMULATED",
    "manual": "MANUAL INPUT",
    "csv": "RECORDED",
    "serial": "LIVE HARDWARE",
}


def _init_state() -> None:
    """Provide defaults on the very first run (and after a browser reset)."""
    st.session_state.setdefault("expected", dict(DEFAULT_EXPECTED))
    st.session_state.setdefault("tolerances", dict(DEFAULT_TOLERANCES))
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_runs", {})
    st.session_state.setdefault("last_qualities", {})
    st.session_state.setdefault("history_source_filter", "All sources")
    st.session_state.setdefault("confirm_clear", False)
    for (group, key), widget_key in _CFG_KEYS.items():
        st.session_state.setdefault(widget_key, st.session_state[group][key])
    # Data-source state -----------------------------------------------------
    st.session_state.setdefault("source_mode", "simulation")
    st.session_state.setdefault("simulator", None)
    st.session_state.setdefault("chart_points", [])
    st.session_state.setdefault("csv_records", None)
    st.session_state.setdefault("csv_errors", [])
    st.session_state.setdefault("csv_warnings", [])
    st.session_state.setdefault("csv_name", "")
    st.session_state.setdefault("csv_index", 0)
    st.session_state.setdefault("serial_reader", None)
    st.session_state.setdefault("serial_ports", None)
    st.session_state.setdefault("serial_connected_device", None)
    st.session_state.setdefault("serial_last_valid_at", None)
    st.session_state.setdefault("serial_malformed", 0)
    st.session_state.setdefault("serial_last_errors", [])


def _restore_defaults() -> None:
    """Reset the configuration form and the active configuration to the
    synthetic demo defaults. Used as a button callback."""
    for (group, key), widget_key in _CFG_KEYS.items():
        defaults = DEFAULT_EXPECTED if group == "expected" else DEFAULT_TOLERANCES
        st.session_state[widget_key] = defaults[key]
    st.session_state["expected"] = dict(DEFAULT_EXPECTED)
    st.session_state["tolerances"] = dict(DEFAULT_TOLERANCES)


_init_state()

# --------------------------------------------------------------------------- #
# Small display helpers
# --------------------------------------------------------------------------- #

STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    # status label: (background, foreground, icon)
    "NORMAL": ("#062b17", "#7ee2a8", "\u2705"),
    "INTEGRITY ANOMALY": ("#3f0d0d", "#f7a8a8", "\U0001f6a8"),
    "INSUFFICIENT DATA": ("#3a2405", "#f5d47e", "\u26a0\ufe0f"),
    "NOT RUN YET": ("#131a24", "#9fb0c8", "\u25cb"),
}

QUALITY_STYLE: dict[str, tuple[str, str, str]] = {
    # quality flag: (background, foreground, label)
    "ok": ("#062b17", "#7ee2a8", "OK"),
    "stale": ("#3a2405", "#f5d47e", "STALE"),
    "missing": ("#3f0d0d", "#f7a8a8", "MISSING"),
    "invalid": ("#3f0d0d", "#f7a8a8", "INVALID"),
    "unknown": ("#131a24", "#9fb0c8", "UNKNOWN"),
}


def _status_banner_html(status_label: str) -> str:
    background, foreground, icon = STATUS_STYLE[status_label]
    return (
        f'<div class="cte-banner" style="background:{background};'
        f'border-color:{foreground}55;">'
        f'<div class="cte-kicker" style="color:{foreground}99;">'
        f"Current verification status</div>"
        f'<div class="cte-status" style="color:{foreground};">'
        f"{icon} {status_label}</div></div>"
    )


def _source_chip(mode: str, connected: bool | None = None) -> str:
    """Honest data-source chip: never says LIVE HARDWARE unless connected."""
    if mode == "serial" and not connected:
        label, bg, fg = "SERIAL \u2014 NOT CONNECTED", "#131a24", "#9fb0c8"
    else:
        label = SOURCE_LABELS[mode]
        bg, fg = {
            "SIMULATED": ("#0d2340", "#7fb3ff"),
            "MANUAL INPUT": ("#3a2405", "#f5d47e"),
            "RECORDED": ("#241a3d", "#c3a6ff"),
            "LIVE HARDWARE": ("#062b17", "#7ee2a8"),
        }[label]
    return (
        f'<span class="cte-chip" style="background:{bg};color:{fg};'
        f'border-color:{fg}55;">{label}</span>'
    )


def _quality_chips_html(quality: dict[str, str] | None) -> str:
    """Per-measurement data-quality chips."""
    if not quality:
        return ""
    chips = []
    for key in MEASUREMENT_KEYS:
        flag = str(quality.get(key, "unknown")).lower()
        bg, fg, label = QUALITY_STYLE.get(flag, QUALITY_STYLE["unknown"])
        chips.append(
            f'<span class="cte-chip" style="background:{bg};color:{fg};'
            f'border-color:{fg}55;">{LABELS[key]}: {label}</span>'
        )
    return "".join(chips)


def _fmt(value: Any, key: str) -> str:
    """Format a measurement value for the dashboard ('-' when unusable)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return "-"
    return f"{number:.{DECIMALS[key]}f}"


def _delta_for(record: dict[str, Any], key: str) -> tuple[str | None, str]:
    """Return (delta text, delta colour) for an observed-value metric card.

    Gray means 'inside tolerance'; red is used whenever the residual is
    beyond tolerance, regardless of the sign.
    """
    residual = record["residuals"].get(key)
    if residual is None:
        return None, "off"
    delta_text = f"{residual:+.{DECIMALS[key]}f} vs expected"
    within = record["passed"].get(key)
    if within in (None, True):
        colour = "off"
    else:
        colour = "inverse" if residual >= 0 else "normal"
    return delta_text, colour


def _pass_label(value: bool | None) -> str:
    if value is None:
        return "NOT CHECKED"
    return "PASS" if value else "FAIL"


# --------------------------------------------------------------------------- #
# Shared verification + charting pipeline
# --------------------------------------------------------------------------- #

def _record_run(
    *,
    source: str,
    scenario_id: str,
    scenario_name: str,
    observed: dict[str, Any],
    quality: dict[str, str] | None,
    claimed_state: str,
    expected_override: dict[str, float] | None,
    sample_time: str | None,
    tolerances_override: dict[str, float] | None = None,
    reference_source: str | None = None,
) -> dict[str, Any]:
    """Verify one sample, store it in history and remember it for display.

    This is the single funnel every data source uses, so simulation, manual
    entry, CSV replay and serial input all get identical verification
    behaviour. ``expected_override`` and ``tolerances_override`` let a data
    source supply its own comparison parameters (manual entry does); when
    they are None the Configuration-page values are used.
    ``reference_source`` records, for display, where the expected values /
    tolerances came from when the user explicitly chose that (manual entry);
    it is None when the Configuration-page values were used implicitly.
    """
    if tolerances_override is not None:
        tolerances = dict(tolerances_override)
    else:
        tolerances = st.session_state["tolerances"]
    if expected_override is not None:
        expected = dict(expected_override)
    else:
        expected = dict(st.session_state["expected"])

    result = verify(
        observed,
        expected,
        tolerances,
        claimed_state,
        data_quality=quality,
    )
    history = st.session_state["history"]
    record = build_record(
        run=next_run_number(history),
        timestamp=datetime.now().astimezone(),
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        claimed_state=claimed_state,
        observed=observed,
        expected=expected,
        tolerances=tolerances,
        result=result,
        source=source,
        reference_source=reference_source,
        sample_time=sample_time,
    )
    stored = append_record(history, record)
    st.session_state["last_runs"][source] = stored
    st.session_state["last_qualities"][source] = (
        dict(quality) if quality else None
    )
    st.session_state["chart_points"].append(
        {
            "run": stored["run"],
            "observed": dict(stored["observed"]),
            "expected": dict(stored["expected"]),
            "source": stored["source"],
        }
    )
    return stored


def _render_trend_charts(source: str) -> None:
    """Chart observed vs expected values for one data source's runs.

    Only points produced by ``source`` are plotted, so a manual-input chart
    never shows simulator numbers and vice versa.
    """
    points = [
        p for p in st.session_state["chart_points"] if p.get("source") == source
    ]
    if not points:
        return
    st.subheader("Measurements across samples")
    st.caption(
        f"One point per {source} verification run in this session. "
        "Missing readings appear as gaps. Session-only data: it resets "
        "when the tab closes."
    )
    cols = st.columns(3)
    for col, key in zip(cols, MEASUREMENT_KEYS):
        with col:
            runs = [p["run"] for p in points]
            observed = [p["observed"].get(key) for p in points]
            expected = [p["expected"].get(key) for p in points]
            if HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=runs, y=expected, mode="lines", name="Expected",
                        line=dict(dash="dash", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=runs, y=observed, mode="lines+markers",
                        name="Observed", line=dict(width=2),
                    )
                )
                fig.update_layout(
                    title=f"{LABELS[key]} ({UNITS[key]})",
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=1.14),
                    template="plotly_dark",
                    xaxis_title="Run #",
                )
                st.plotly_chart(fig, theme=None)
            else:  # pragma: no cover - fallback when Plotly is missing
                st.line_chart(
                    pd.DataFrame(
                        {"Expected": expected, "Observed": observed},
                        index=runs,
                    )
                )


def _render_result(record: dict[str, Any], quality: dict[str, str] | None) -> None:
    """Status cards, reasons, comparison table and the attack disclaimer.

    ``quality`` is passed explicitly so each data source shows only its own
    data-quality flags (None for sources that do not provide them).
    """
    source = record.get("source", "SIMULATION")

    header_bits = [
        f"Run #{record['run']}",
        record["scenario_name"],
        f"source: {source}",
        f"verified at {record['timestamp']}",
    ]
    if record.get("sample_time"):
        header_bits.append(f"sample time: {record['sample_time']}")
    st.subheader(" \u2022 ".join(header_bits))
    if record.get("reference_source"):
        st.caption(f"Reference values: {record['reference_source']}")

    if quality:
        st.markdown(
            _quality_chips_html(quality), unsafe_allow_html=True
        )

    # Row 1: claim, status, pH
    cols = st.columns(4)
    cols[0].metric("Controller-claimed state", record["claimed_state"])
    cols[1].markdown(
        _status_banner_html(record["result"]), unsafe_allow_html=True
    )
    cols[2].metric("Expected pH", _fmt(record["expected"]["ph"], "ph"))
    delta, colour = _delta_for(record, "ph")
    cols[3].metric(
        "Observed pH",
        _fmt(record["observed"]["ph"], "ph"),
        delta=delta,
        delta_color=colour,
        help="Delta shows the residual (observed - expected). Gray = within "
             "tolerance, red = beyond tolerance.",
    )

    # Row 2: conductivity and temperature
    cols = st.columns(4)
    cols[0].metric(
        "Expected conductivity",
        f"{_fmt(record['expected']['conductivity'], 'conductivity')} \u00b5S/cm",
    )
    delta, colour = _delta_for(record, "conductivity")
    cols[1].metric(
        "Observed conductivity",
        f"{_fmt(record['observed']['conductivity'], 'conductivity')} "
        f"\u00b5S/cm",
        delta=delta,
        delta_color=colour,
    )
    cols[2].metric(
        "Expected temperature",
        f"{_fmt(record['expected']['temperature'], 'temperature')} \u00b0C",
    )
    delta, colour = _delta_for(record, "temperature")
    cols[3].metric(
        "Observed temperature",
        f"{_fmt(record['observed']['temperature'], 'temperature')} \u00b0C",
        delta=delta,
        delta_color=colour,
    )

    st.markdown(
        '<div class="cte-note">'
        f"<strong>Why did this happen?</strong><br>{record['summary']}"
        "</div>",
        unsafe_allow_html=True,
    )
    for reason in record["reasons"]:
        st.markdown(f"- {reason}")

    # Comparison table: expected / observed / residual / tolerance / quality
    table_rows = []
    for key in MEASUREMENT_KEYS:
        flag = (quality or {}).get(key, "unknown")
        table_rows.append(
            {
                "Measurement": LABELS[key],
                "Unit": UNITS[key],
                "Expected": _fmt(record["expected"][key], key),
                "Observed": _fmt(record["observed"][key], key),
                "Residual (observed - expected)": _fmt(
                    record["residuals"].get(key), key
                ),
                "Tolerance (+/-)": _fmt(record["tolerances"][key], key),
                "Quality": str(flag).upper(),
                "Check": _pass_label(record["passed"].get(key)),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True)

    st.warning(
        "**Reminder:** an anomaly is not proof of a cyberattack. It only "
        "means the numbers disagree beyond the configured tolerance. "
        "Possible explanations include sensor faults, noise, calibration "
        "errors, environmental variation, communication problems or "
        "manipulated data. Deciding which one applies needs investigation "
        "outside this engine.",
        icon="\u26a0\ufe0f",
    )


# --------------------------------------------------------------------------- #
# Data-source panels
# --------------------------------------------------------------------------- #

def _panel_simulation() -> None:
    """Stateful one-sample-at-a-time synthetic simulation."""
    labels = {meta["name"]: sid for sid, meta in SCENARIOS.items()}
    selected_label = st.selectbox("Verification scenario", list(labels))
    scenario_id = labels[selected_label]
    meta = SCENARIOS[scenario_id]

    st.caption(f"**Scenario:** {meta['name']} \u2014 {meta['hint']}")
    st.caption(
        "\U0001f50d The cause described above is known only because the "
        "simulator deliberately injects it. In a real system the detection "
        "engine cannot infer the cause of a mismatch \u2014 it can only "
        "compare numbers."
    )

    expected = st.session_state["expected"]
    tolerances = st.session_state["tolerances"]
    with st.expander("Configuration used for this run"):
        rows = [
            {
                "Measurement": LABELS[key],
                "Unit": UNITS[key],
                "Expected": _fmt(expected[key], key),
                "Tolerance (+/-)": _fmt(tolerances[key], key),
            }
            for key in MEASUREMENT_KEYS
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(
            "Edit these values on the **Configuration** page. They are "
            "synthetic software-demo defaults, not calibrated values."
        )

    col_adv, col_rst, _ = st.columns([1, 1, 2])
    advance = col_adv.button(
        "Advance one sample", type="primary", icon="\u25b6\ufe0f"
    )
    restart = col_rst.button("Restart simulation", icon="\U0001f504")

    if restart:
        st.session_state["simulator"] = None
        st.session_state["chart_points"] = [
            p
            for p in st.session_state["chart_points"]
            if p.get("source") != "SIMULATION"
        ]
        st.info(
            "Simulation restarted. Event history and other data sources' "
            "charts were kept."
        )

    if advance:
        sim = st.session_state.get("simulator")
        # Recreate only when the scenario or seed changed - never on a
        # plain Streamlit rerun.
        if (
            sim is None
            or sim.config.scenario_id != scenario_id
            or sim.tolerances != tolerances
        ):
            sim = make_scenario_simulator(
                scenario_id, seed=DEFAULT_SEED, tolerances=tolerances
            )
            st.session_state["simulator"] = sim
        sample = sim.step()
        _record_run(
            source="SIMULATION",
            scenario_id=scenario_id,
            scenario_name=f"{meta['name']} \u00b7 sample {sample.index}",
            observed=dict(sample.observed),
            quality=dict(sample.quality),
            claimed_state=sample.claimed_state,
            expected_override=sample.expected,
            sample_time=sample.timestamp.strftime("%Y-%m-%d %H:%M:%S (sim clock)"),
        )

    sim = st.session_state.get("simulator")
    if sim is not None and sim.samples:
        st.caption(
            f"Simulator state: sample #{sim.index} on the fixed simulated "
            f"clock (deterministic for seed {sim.config.seed}). "
            "The simulated clock is not real time."
        )


#: Reference-value sources offered in MANUAL INPUT mode. The labels are
#: static (no embedded numbers) so a stored selection never goes stale
#: when Configuration values change.
REF_TYPE_MYSELF = "Enter expected values and tolerances myself"
REF_CONFIG = "Configuration page values (expected + tolerances)"
REF_PROFILE_PREFIX = "Illustrative profile: "  # followed by the profile label


def _reference_options() -> list[str]:
    """Manual-mode reference sources offered to the user (static labels)."""
    options = [REF_TYPE_MYSELF, REF_CONFIG]
    for profile in PROFILES.values():
        options.append(
            f"{REF_PROFILE_PREFIX}{profile.label} "
            "(expected; tolerances from Configuration page)"
        )
    return options


def _reference_choice_map() -> dict[str, str]:
    """Option label -> profile state_id ('' for the two non-profile ones)."""
    mapping = {REF_TYPE_MYSELF: "", REF_CONFIG: ""}
    for state_id, profile in PROFILES.items():
        mapping[
            f"{REF_PROFILE_PREFIX}{profile.label} "
            "(expected; tolerances from Configuration page)"
        ] = state_id
    return mapping


def _resolve_reference(
    choice: str,
) -> tuple[dict[str, float] | None, dict[str, float] | None, str]:
    """Resolve the six reference fields for an explicitly selected source.

    Returns ``(expected, tolerances, provenance label)``. Both mappings are
    None when nothing usable was selected: a missing reference is never
    substituted with demonstration values.
    """
    if choice == REF_CONFIG:
        return (
            dict(st.session_state["expected"]),
            dict(st.session_state["tolerances"]),
            "Configuration page values",
        )
    state_id = _reference_choice_map().get(choice)
    if state_id:  # an illustrative profile
        profile = PROFILES[state_id]
        return (
            profile.expected_values(),
            dict(st.session_state["tolerances"]),
            (
                f"illustrative profile \u2018{profile.label}\u2019 "
                "(expected) + Configuration page (tolerances)"
            ),
        )
    return None, None, "typed into the form"


def _reference_text(value: float | None) -> str:
    """Format a reference number for the manual text-input fields.

    ``None`` (no reference available) becomes an empty string so the field
    stays blank; numbers keep enough precision to round-trip through
    ``float()`` when the form is submitted.
    """
    if value is None:
        return ""
    return str(value)


def _apply_reference_source() -> None:
    """Fill or clear the six reference fields for the selected source.

    Runs as the reference selector's on_change callback, i.e. before the
    form renders in the new run - the only Streamlit-safe moment to write
    another widget's key. Observed readings are never touched here.

    The reference fields are text inputs, so values are written as strings
    (empty string = cleared); ``parse_manual_entry`` coerces them back to
    numbers on submit.
    """
    choice = st.session_state.get("man_ref_source", REF_TYPE_MYSELF)
    expected, tolerances, label = _resolve_reference(choice)
    if expected is None:
        expected = {"ph": None, "conductivity": None, "temperature": None}
        tolerances = {"ph": None, "conductivity": None, "temperature": None}
    st.session_state["man_exp_ph"] = _reference_text(expected["ph"])
    st.session_state["man_exp_cond"] = _reference_text(expected["conductivity"])
    st.session_state["man_exp_temp"] = _reference_text(expected["temperature"])
    st.session_state["man_tol_ph"] = _reference_text(tolerances["ph"])
    st.session_state["man_tol_cond"] = _reference_text(tolerances["conductivity"])
    st.session_state["man_tol_temp"] = _reference_text(tolerances["temperature"])
    st.session_state["man_ref_snapshot"] = (
        None
        if expected["ph"] is None
        else (label, dict(expected), dict(tolerances))
    )


def _submitted_reference_provenance(
    expected: dict[str, float], tolerances: dict[str, float]
) -> str:
    """Label for where the submitted reference values actually came from."""
    snapshot = st.session_state.get("man_ref_snapshot")
    if snapshot is None:
        return "typed into the form"
    label, snap_expected, snap_tolerances = snapshot
    if dict(expected) == snap_expected and dict(tolerances) == snap_tolerances:
        return label
    return f"{label} \u2014 edited in the form"


def _panel_manual() -> None:
    """Readings typed by the user, verified through the existing engine."""
    st.info(
        "MANUAL ENTRY MODE \u2014 Values are entered by the user and are "
        "not automatically verified physical measurements."
    )
    st.caption(
        "How a verification works: the detection engine computes "
        "**residual = observed \u2212 expected** for each measurement and "
        "compares its absolute value with your tolerance. A residual with "
        "`|residual| <= tolerance` passes; anything larger fails. All nine "
        "fields below start empty and nothing is verified until you press "
        "Verify Measurements \u2014 there are no hidden defaults."
    )

    st.selectbox(
        "Expected values and tolerances come from:",
        _reference_options(),
        key="man_ref_source",
        on_change=_apply_reference_source,
        help="Changing this fills or clears the six reference fields "
             "(expected + tolerances) below. Your observed readings are "
             "never touched by it.",
    )
    choice = st.session_state.get("man_ref_source", REF_TYPE_MYSELF)
    expected, tolerances, label = _resolve_reference(choice)
    if expected is None:
        st.caption(
            "Reference source: **typed into the form** \u2014 the six "
            "reference fields below start empty. Missing values are "
            "reported as validation errors; they are never replaced with "
            "demonstration numbers."
        )
    else:
        st.caption(
            f"Reference source: **{label}** \u2014 expected pH "
            f"{expected['ph']:.2f}, {expected['conductivity']:.1f} "
            f"\u00b5S/cm, {expected['temperature']:.2f} \u00b0C; "
            f"tolerances \u00b1{tolerances['ph']:.2f}, "
            f"\u00b1{tolerances['conductivity']:.1f}, "
            f"\u00b1{tolerances['temperature']:.2f} (Configuration "
            "page). The fields below stay editable."
        )

    with st.form("manual_entry_form"):
        st.markdown("#### 1. Controller-reported state")
        claim = st.selectbox(
            "Controller-reported state",
            list(CLAIM_OPTIONS),
            key="man_claim",
            help="This is supporting evidence only - it is never treated "
                 "as proof that the physical process is normal.",
        )
        for option in CLAIM_OPTIONS:
            st.caption(f"- **{option}:** {CLAIM_NOTES[option]}")

        st.markdown("#### 2. Observed measurements (what was measured)")
        cols = st.columns(3)
        obs_ph = cols[0].text_input(
            "Observed pH (dimensionless)",
            key="man_obs_ph",
            placeholder="e.g. 8.20",
            help="Required. Type what was actually measured, 0 to 14. "
                 "The field starts empty on purpose.",
        )
        obs_cond = cols[1].text_input(
            "Observed conductivity (\u00b5S/cm)",
            key="man_obs_cond",
            placeholder="e.g. 650",
            help="Required. Must be 0 or greater (the app rejects "
                 "negative values). The field starts empty on purpose.",
        )
        obs_temp = cols[2].text_input(
            "Observed temperature (\u00b0C)",
            key="man_obs_temp",
            placeholder="e.g. 25.50",
            help="Required. Sensor limits used across this app: -50 to "
                 "200 \u00b0C. The field starts empty on purpose.",
        )

        st.markdown("#### 3. Expected measurements (for the claimed state)")
        cols = st.columns(3)
        exp_ph = cols[0].text_input(
            "Expected pH (dimensionless)",
            key="man_exp_ph",
            placeholder="e.g. 8.00",
        )
        exp_cond = cols[1].text_input(
            "Expected conductivity (\u00b5S/cm)",
            key="man_exp_cond",
            placeholder="e.g. 650",
        )
        exp_temp = cols[2].text_input(
            "Expected temperature (\u00b0C)",
            key="man_exp_temp",
            placeholder="e.g. 25.50",
        )

        st.markdown("#### 4. Acceptable tolerances (\u00b1 maximum deviation)")
        cols = st.columns(3)
        tol_ph = cols[0].text_input(
            "pH tolerance (\u00b1, dimensionless)",
            key="man_tol_ph",
            placeholder="e.g. 0.50",
            help="A measurement passes when |residual| <= tolerance.",
        )
        tol_cond = cols[1].text_input(
            "Conductivity tolerance (\u00b1, \u00b5S/cm)",
            key="man_tol_cond",
            placeholder="e.g. 80",
        )
        tol_temp = cols[2].text_input(
            "Temperature tolerance (\u00b1, \u00b0C)",
            key="man_tol_temp",
            placeholder="e.g. 2.00",
        )

        submitted = st.form_submit_button(
            "Verify Measurements", type="primary"
        )

    if not submitted:
        return

    entry, errors = parse_manual_entry(
        claimed_state=claim,
        observed={
            "ph": obs_ph, "conductivity": obs_cond, "temperature": obs_temp,
        },
        expected={
            "ph": exp_ph, "conductivity": exp_cond, "temperature": exp_temp,
        },
        tolerances={
            "ph": tol_ph, "conductivity": tol_cond, "temperature": tol_temp,
        },
    )
    if errors:
        for message in errors:
            st.error(message)
        st.warning(
            "No verification was recorded: the input was invalid. Invalid "
            "values are never replaced with normal measurements.",
            icon="\u26a0\ufe0f",
        )
        # Nothing was verified: make sure the result area does not keep
        # showing an older MANUAL INPUT result as if this input passed.
        st.session_state["last_runs"].pop("MANUAL INPUT", None)
        st.session_state["last_qualities"].pop("MANUAL INPUT", None)
        return

    reference_source = _submitted_reference_provenance(
        dict(entry.expected), dict(entry.tolerances)
    )
    now = datetime.now().astimezone()
    record = _record_run(
        source="MANUAL INPUT",
        scenario_id="manual_input",
        scenario_name=f"Manual entry \u00b7 {entry.claimed_state}",
        observed=dict(entry.observed),
        quality=None,
        claimed_state=entry.claimed_state,
        expected_override=dict(entry.expected),
        tolerances_override=dict(entry.tolerances),
        reference_source=reference_source,
        sample_time=now.strftime("%Y-%m-%d %H:%M:%S (entered by user)"),
    )
    st.success(
        f"Manual verification recorded as run #{record['run']} "
        f"(source: MANUAL INPUT; reference values: {reference_source}). "
        "The structured result is shown below."
    )
    st.caption(
        "An integrity anomaly indicates a mismatch with the configured "
        "expected response. It does not prove that a cyberattack occurred."
    )


def _panel_csv() -> None:
    """CSV replay of recorded measurements."""
    st.caption(
        "**Recorded data** \u2014 replayed, not verified or experimentally "
        "calibrated. Expected columns: "
        + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
        + " (optional `claimed_state`). "
        "pH dimensionless, conductivity in \u00b5S/cm, temperature in \u00b0C."
    )
    uploaded = st.file_uploader(
        "Upload a CSV of recorded measurements", type=["csv"]
    )

    if uploaded is not None:
        identity = (uploaded.name, getattr(uploaded, "size", None))
        if st.session_state.get("csv_identity") != identity:
            st.session_state["csv_identity"] = identity
            try:
                text = uploaded.getvalue().decode("utf-8-sig")
            except UnicodeDecodeError:
                text = uploaded.getvalue().decode("utf-8", errors="replace")
            result = parse_csv(text)
            st.session_state["csv_records"] = result.records
            st.session_state["csv_errors"] = list(result.errors)
            st.session_state["csv_warnings"] = list(result.warnings)
            st.session_state["csv_name"] = uploaded.name
            st.session_state["csv_index"] = 0

    records = st.session_state.get("csv_records")
    if records is None:
        st.info(
            "No CSV loaded yet. Upload a file to replay it one sample at a "
            "time. Nothing is read from disk or sent anywhere \u2014 the "
            "file stays in this browser session only."
        )
        return

    errors = st.session_state["csv_errors"]
    warnings = st.session_state["csv_warnings"]
    total = len(records)
    done = st.session_state["csv_index"]

    if errors:
        with st.expander(
            f"\u26a0\ufe0f {len(errors)} validation problem(s) found "
            "(invalid rows are replayed as INSUFFICIENT DATA, never as "
            "normal values)",
            expanded=False,
        ):
            for message in errors:
                st.markdown(f"- {message}")
    if warnings:
        with st.expander(f"\U0001f4a1 {len(warnings)} warning(s)", expanded=False):
            for message in warnings:
                st.markdown(f"- {message}")

    st.caption(
        f"File: **{st.session_state['csv_name']}** \u2014 "
        f"{total} usable row(s); replay position {min(done, total)}/{total}."
    )

    col_next, col_rst, _ = st.columns([1, 1, 2])
    advance = col_next.button(
        "Replay next sample", type="primary", icon="\u25b6\ufe0f"
    )
    restart = col_rst.button("Restart replay", icon="\U0001f504")

    if restart:
        st.session_state["csv_index"] = 0
        st.info("Replay rewound to the first row. Event history was kept.")

    if advance:
        index = st.session_state["csv_index"]
        if index >= total:
            st.warning("End of file reached. Restart replay to begin again.")
        else:
            record = records[index]
            st.session_state["csv_index"] = index + 1
            sample_time = (
                record.timestamp.strftime("%Y-%m-%d %H:%M:%S (file)")
                if record.timestamp is not None
                else "unparseable timestamp in file"
            )
            _record_run(
                source="CSV REPLAY",
                scenario_id="csv_replay",
                scenario_name=(
                    f"CSV row {record.row_number} "
                    f"({st.session_state['csv_name']})"
                ),
                observed=dict(record.observed),
                quality=dict(record.quality),
                claimed_state=record.claimed_state,
                expected_override=None,
                sample_time=sample_time,
            )


def _panel_serial() -> None:
    """Optional, read-only ESP32 USB serial input."""
    st.caption(
        "\U0001f6e1\ufe0f **Read-only by design.** This mode never sends "
        "commands to the device and never operates pumps, relays or dosing "
        "equipment. Nothing connects automatically \u2014 you choose a port "
        "and press Connect."
    )
    st.caption(
        "\U0001f310 **Hosted deployments cannot use serial mode.** A remotely "
        "hosted copy (for example on Streamlit Community Cloud) runs in a "
        "remote data centre and cannot access a USB-connected ESP32 on your "
        "computer - no ports will be detected there. Serial mode only works "
        "when you run this app locally on the machine the device is plugged "
        "into. Simulation, manual input and CSV replay work everywhere."
    )

    if not pyserial_available():
        st.info(
            "pySerial is not installed, so ESP32 USB serial input is "
            "unavailable. Simulation and CSV replay still work fully. To "
            "enable serial input, run: `python -m pip install pyserial`"
        )
        return

    reader: SerialReader | None = st.session_state.get("serial_reader")
    connected = bool(reader and reader.connected)

    # ---- port selection -------------------------------------------------
    col_sel, col_ref = st.columns([3, 1])
    if col_ref.button("Refresh ports", icon="\U0001f504"):
        ports, err = list_ports()
        st.session_state["serial_ports"] = (ports, err)
        st.session_state.pop("serial_selected_port", None)
    if st.session_state.get("serial_ports") is None:
        ports, err = list_ports()
        st.session_state["serial_ports"] = (ports, err)
    ports, list_err = st.session_state["serial_ports"]
    if list_err:
        st.warning(list_err)

    port_options = [
        f"{p.device} \u2014 {p.description}" if p.description else p.device
        for p in ports
    ]
    if port_options and not connected:
        selected = col_sel.selectbox(
            "Detected serial ports", port_options, key="serial_selected_port"
        )
    elif connected:
        col_sel.markdown(
            f"**Connected port:** `{st.session_state['serial_connected_device']}`"
        )
    else:
        col_sel.warning(
            "No serial ports detected. Check the USB cable and driver, then "
            "press Refresh ports."
        )

    # ---- connect / disconnect -------------------------------------------
    baud = st.number_input(
        "Baud rate",
        min_value=1200,
        max_value=921600,
        value=DEFAULT_BAUDRATE,
        step=1200,
        key="serial_baud",
    )
    col_c, col_d, _ = st.columns([1, 1, 2])
    connect_clicked = col_c.button(
        "Connect", type="primary", disabled=connected or not port_options
    )
    disconnect_clicked = col_d.button("Disconnect", disabled=not connected)

    if disconnect_clicked and reader is not None:
        reader.disconnect()
        st.session_state["serial_connected_device"] = None
        st.info("Disconnected.")
    if connect_clicked:
        device = selected.split(" \u2014 ")[0]
        if reader is None:
            reader = SerialReader()
            st.session_state["serial_reader"] = reader
        error = reader.connect(device, baudrate=baud)
        if error:
            st.error(error)
        else:
            st.session_state["serial_connected_device"] = device
            st.success(f"Connected to {device} (read-only).")

    # ---- freshness controls + reading -----------------------------------
    stale_limit = st.number_input(
        "Freshness limit (seconds)",
        min_value=1.0,
        max_value=600.0,
        value=DEFAULT_STALENESS_SECONDS,
        step=1.0,
        key="serial_stale_limit",
        help="A sample older than this is treated as stale data "
             "(INSUFFICIENT DATA), not as a normal reading.",
    )

    if reader is not None and reader.connected:
        now = datetime.now().astimezone()
        fresh = not is_stale(reader.last_valid_at, now, stale_limit)
        age = reader.last_valid_age_seconds
        if reader.last_valid_at is None:
            st.caption(
                "No valid sample received yet. Press **Read next sample**."
            )
        elif fresh:
            st.caption(
                f"\u2705 Freshness: last valid sample "
                f"{age:.1f} s ago (limit {stale_limit:.0f} s)."
            )
        else:
            st.warning(
                f"\u26a0\ufe0f STALE \u2014 the last valid sample is "
                f"{age:.1f} s old (limit {stale_limit:.0f} s). Any "
                "verification of this data yields INSUFFICIENT DATA."
            )

        if st.button("Read next sample", type="primary", icon="\u25b6\ufe0f"):
            message, errors = reader.read_message(
                default_claimed_state="Steady operation"
            )
            st.session_state["serial_malformed"] = reader.malformed_count
            st.session_state["serial_last_errors"] = list(errors)
            if message is None:
                for error in errors:
                    st.error(error)
            else:
                st.session_state["serial_last_valid_at"] = now
                sample_time = (
                    message.timestamp.strftime("%Y-%m-%d %H:%M:%S (device)")
                    if message.timestamp is not None
                    else f"received {now.strftime('%H:%M:%S')} (no device timestamp)"
                )
                _record_run(
                    source="SERIAL",
                    scenario_id="serial_sample",
                    scenario_name="ESP32 serial sample",
                    observed=dict(message.observed),
                    quality=dict(message.quality),
                    claimed_state=message.claimed_state or "Steady operation",
                    expected_override=None,
                    sample_time=sample_time,
                )

        st.caption(
            f"Malformed messages rejected so far: "
            f"**{st.session_state['serial_malformed']}**. "
            "Every message is validated; malformed lines are never guessed "
            "at."
        )
    elif reader is None or not reader.connected:
        st.info(
            "Serial mode is idle. Select a port and press **Connect** to "
            "read from an ESP32 that streams the documented JSON lines. The "
            "app was developed and tested WITHOUT hardware, so treat live "
            "serial reads as untested until you have verified them."
        )


# --------------------------------------------------------------------------- #
# Page renderers
# --------------------------------------------------------------------------- #

def render_overview() -> None:
    st.title("Chemical Truth Engine 2.0")
    st.caption(
        "Physics-Informed Integrity Verification for Chemical "
        "Cyber-Physical Systems"
    )
    st.warning(
        "**EDUCATIONAL SIMULATION - NO REAL CHEMICAL PROCESS OR HARDWARE IS "
        "CONNECTED.** Unless you explicitly connect an ESP32 in serial "
        "mode, every value in this application is generated by software. "
        "The default numbers are synthetic examples, not experimental "
        "measurements."
    )

    st.markdown(
        """
        ### The idea in one sentence
        > *"The computer tells us what happened; chemistry helps us check "
        > whether that claim makes physical sense."*

        A normal monitoring dashboard shows readings. This prototype goes one
        step further: it compares the **controller's claimed state** and the
        **observed physical measurements** against an **expected response**
        (a synthetic, configurable model of how the process should behave in
        that state). When observations disagree with the expectation beyond a
        configured tolerance, the system raises an **integrity anomaly**.

        The software clearly separates four things:

        1. What the controller **claims** happened.
        2. What the sensors **observed** (simulated, replayed or serial).
        3. What **should** have been observed if the claim were true.
        4. The **comparison result** produced by the detection engine.
        """
    )

    st.markdown(
        """
        ### Four ways to feed it data

        | Mode | What it is | Honesty label |
        |---|---|---|
        | **Simulation** | A stateful, deterministic time-series simulator you advance one sample at a time. Works immediately with no hardware. | `SIMULATED` |
        | **Manual input** | Type observed readings, expected values and tolerances into a form; the engine verifies them. | `MANUAL INPUT` |
        | **CSV replay** | Upload a file of recorded measurements; replay it row by row with full validation. | `RECORDED` |
        | **ESP32 USB serial** | Optional, read-only serial input from real hardware you explicitly connect. Never sends commands. | `LIVE HARDWARE` |
        """
    )

    diagram = """
    +----------------------+   +--------------------------+   +---------------------+
    | Controller claim     |   | Expected response        |   | Observed readings   |
    | "Target reached"     |   | (profiles / config)      |   | (sim/manual/CSV/ser)|
    +----------+-----------+   +-------------+------------+   +----------+----------+
               |                             |                           |
               +-------------+---------------+---------------------------+
                                             |
                                             v
                        +--------------------------------------------+
                        | detection_engine.py                        |
                        | residual = observed - expected             |
                        | pass if |residual| <= tolerance            |
                        +--------------------+-----------------------+
                                             |
                                             v
        NORMAL   |   INTEGRITY ANOMALY   |   INSUFFICIENT DATA
    """
    st.markdown(
        f'<div class="cte-diagram"><pre style="margin:0;">{diagram}</pre></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### How to read the results

        | Status | Meaning |
        |---|---|
        | **NORMAL** | Every valid measurement agrees with the expected response within its tolerance. |
        | **INTEGRITY ANOMALY** | At least one valid measurement deviates beyond its tolerance. This is a discrepancy, **not proof of a cyberattack** - sensor faults, noise, calibration errors, environment or manipulated data can look identical. |
        | **INSUFFICIENT DATA** | Required information was missing, invalid or stale, so no conclusion was drawn. Missing data never counts as a pass, and it is not automatically an attack. |

        ### Try it

        1. Open **Demonstration** in the sidebar (mode: Simulation).
        2. Pick a scenario and press **Advance one sample** a few times.
        3. Read the status card, the comparison table and the reasons.
        4. Watch the trend chart fill in across samples.
        5. Switch to MANUAL INPUT, choose a reference source, type your own
           readings and press Verify Measurements - nothing runs until you
           submit.
        6. Open **History & Charts** for the event log and per-run trends.
        """
    )


def render_demo() -> None:
    st.title("Demonstration and verification")
    st.caption(
        "Produce a sample and verify it. Nothing is generated by ordinary "
        "page reruns \u2014 only by the buttons you press."
    )

    mode_label = st.radio(
        "Data source",
        list(SOURCE_MODES),
        key="source_mode_radio",
        horizontal=True,
    )
    mode = SOURCE_MODES[mode_label]

    connected = bool(
        st.session_state.get("serial_reader")
        and st.session_state["serial_reader"].connected
    )
    st.markdown(
        _source_chip(mode, connected=connected), unsafe_allow_html=True
    )
    if mode == "csv":
        st.caption(
            "Imported rows are **recorded data** \u2014 replayed and "
            "validated for structure, but not verified or experimentally "
            "calibrated."
        )

    st.divider()

    if mode == "simulation":
        _panel_simulation()
    elif mode == "manual":
        _panel_manual()
    elif mode == "csv":
        _panel_csv()
    else:
        _panel_serial()

    st.divider()

    record_source = {
        "simulation": "SIMULATION",
        "manual": "MANUAL INPUT",
        "csv": "CSV REPLAY",
        "serial": "SERIAL",
    }[mode]
    empty_states = {
        "simulation": (
            "No verification has been run in this session yet. Use the "
            "controls above to produce a sample."
        ),
        "manual": (
            "No manual verification yet in this session. Choose a "
            "reference source and fill in the nine fields above, then "
            "press Verify Measurements. Nothing is verified or recorded "
            "until you submit - the app never runs manual checks on its "
            "own."
        ),
        "csv": (
            "No CSV row replayed yet in this session. Upload a file and "
            "press Replay next sample."
        ),
        "serial": (
            "No serial sample read yet in this session. Connect to a "
            "port and press Read next sample (read-only)."
        ),
    }

    record = st.session_state["last_runs"].get(record_source)
    if record is None:
        st.info(empty_states[mode])
        return

    _render_result(record, st.session_state["last_qualities"].get(record_source))
    _render_trend_charts(record_source)


def render_config() -> None:
    st.title("Expected-response configuration")
    st.warning(
        "The values below are **synthetic software-demo defaults only**. "
        "They are not scientifically calibrated, not measured in a "
        "laboratory, not validated for any real process and not universally "
        "valid.",
        icon="\u26a0\ufe0f",
    )
    st.caption(
        "A real implementation would require experimental calibration under "
        "defined process conditions, with the measurement uncertainty of "
        "each instrument considered. Note also that pH must never be assumed "
        "to vary linearly with chemical dose - no such reaction model is "
        "used or claimed to be verified anywhere in this prototype. These "
        "values feed SIMULATION, CSV REPLAY and SERIAL runs. In MANUAL "
        "INPUT mode the expected values and tolerances come only from an "
        "explicitly selected source (typed by you, these settings, or an "
        "illustrative profile) - never silently."
    )

    with st.form("config_form"):
        st.markdown("#### Expected response (for the claimed state)")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Expected pH (dimensionless)",
                min_value=0.0, max_value=14.0, step=0.1, format="%.2f",
                key="cfg_exp_ph",
                help="Illustrative default: 8.00",
            )
            st.number_input(
                "Expected conductivity (\u00b5S/cm)",
                min_value=0.0, max_value=100_000.0, step=1.0, format="%.1f",
                key="cfg_exp_conductivity",
                help="Illustrative default: 650.0",
            )
            st.number_input(
                "Expected temperature (\u00b0C)",
                min_value=-50.0, max_value=200.0, step=0.1, format="%.2f",
                key="cfg_exp_temperature",
                help="Illustrative default: 25.50",
            )
        with col2:
            st.markdown("#### Tolerances (maximum acceptable deviation)")
            st.number_input(
                "pH tolerance (+/-, dimensionless)",
                min_value=0.0, max_value=14.0, step=0.05, format="%.2f",
                key="cfg_tol_ph",
                help="Illustrative default: +/-0.50. A measurement passes "
                     "when |residual| <= tolerance.",
            )
            st.number_input(
                "Conductivity tolerance (+/-, \u00b5S/cm)",
                min_value=0.0, max_value=100_000.0, step=1.0, format="%.1f",
                key="cfg_tol_conductivity",
                help="Illustrative default: +/-80.0",
            )
            st.number_input(
                "Temperature tolerance (+/-, \u00b0C)",
                min_value=0.0, max_value=250.0, step=0.1, format="%.2f",
                key="cfg_tol_temperature",
                help="Illustrative default: +/-2.00",
            )
        submitted = st.form_submit_button(
            "Apply configuration", type="primary"
        )

    if submitted:
        new_expected = {
            "ph": st.session_state["cfg_exp_ph"],
            "conductivity": st.session_state["cfg_exp_conductivity"],
            "temperature": st.session_state["cfg_exp_temperature"],
        }
        new_tolerances = {
            "ph": st.session_state["cfg_tol_ph"],
            "conductivity": st.session_state["cfg_tol_conductivity"],
            "temperature": st.session_state["cfg_tol_temperature"],
        }
        issues = validate_configuration(new_expected, new_tolerances)
        if issues:
            for issue in issues:
                st.error(issue)
            st.error(
                "Configuration **not applied**. Fix the values above and "
                "apply again - nothing was silently substituted."
            )
        else:
            st.session_state["expected"] = dict(new_expected)
            st.session_state["tolerances"] = dict(new_tolerances)
            st.success(
                "Configuration applied. It affects the **next** verification "
                "run."
            )

    st.button(
        "Restore synthetic demo defaults",
        on_click=_restore_defaults,
        icon="\U0001f504",  # 🔄
    )

    st.divider()
    st.subheader("Active configuration")
    active_rows = [
        {
            "Measurement": LABELS[key],
            "Unit": UNITS[key],
            "Expected": _fmt(st.session_state["expected"][key], key),
            "Tolerance (+/-)": _fmt(st.session_state["tolerances"][key], key),
        }
        for key in MEASUREMENT_KEYS
    ]
    st.dataframe(pd.DataFrame(active_rows), hide_index=True)
    st.caption(
        "Changing the tolerance changes later results: a wider tolerance "
        "accepts larger deviations, a narrower one accepts fewer."
    )

    st.divider()
    st.subheader("Illustrative process profiles")
    st.caption(
        "These profiles define the expected response for each claimed "
        "process state. They are invented demonstration values - NOT "
        "measured, calibrated or scientifically validated. Real calibration "
        "data would replace them in process_profiles.py without any other "
        "change."
    )
    profile_rows = [
        {
            "State": p.label,
            "pH": f"{p.ph:.2f}",
            "Conductivity (\u00b5S/cm)": f"{p.conductivity:.1f}",
            "Temperature (\u00b0C)": f"{p.temperature:.2f}",
            "Note": p.description,
        }
        for p in PROFILES.values()
    ]
    st.dataframe(pd.DataFrame(profile_rows), hide_index=True)


def render_history() -> None:
    st.title("Verification history and trends")
    history = st.session_state["history"]

    # ------------------------------------------------------------------ #
    # Clear controls (two-step confirmation)
    # ------------------------------------------------------------------ #
    if st.button("Clear demo history", icon="\U0001f5d1\ufe0f"):
        st.session_state["confirm_clear"] = True
    if st.session_state.get("confirm_clear"):
        st.warning(
            "This removes every recorded run and the current result from "
            "this browser session. Configuration values are kept. "
            "Continue?",
            icon="\u26a0\ufe0f",
        )
        col_yes, col_no, _ = st.columns([1, 1, 3])
        if col_yes.button("Yes, clear everything", type="primary"):
            clear_history(history)
            st.session_state["last_runs"] = {}
            st.session_state["last_qualities"] = {}
            st.session_state["chart_points"] = []
            st.session_state["history_source_filter"] = "All sources"
            st.session_state["confirm_clear"] = False
            st.rerun()
        if col_no.button("Cancel"):
            st.session_state["confirm_clear"] = False
            st.rerun()

    if not history:
        st.info(
            "The event history is empty. Run a few verifications on the "
            "**Demonstration** page and they will appear here.\n\n"
            "History is stored only in this browser session "
            "(`st.session_state`), so it is temporary and disappears when "
            "the tab is closed or Streamlit restarts."
        )
        return

    st.caption(
        f"{len(history)} run(s) recorded. Timestamps use this computer's "
        "local time. Source shows whether each run came from the simulator, "
        "your manual entry, a CSV file or the serial port; Reference shows "
        "where the expected values / tolerances came from when that was an "
        "explicit choice (manual entry)."
    )

    # ------------------------------------------------------------------ #
    # Trend charts: observed vs expected per measurement
    # ------------------------------------------------------------------ #
    st.subheader("Observed vs expected over runs")
    sources_present = list(dict.fromkeys(r.get("source") for r in history))
    filter_options = ["All sources", *sources_present]
    if st.session_state.get("history_source_filter") not in filter_options:
        st.session_state["history_source_filter"] = "All sources"
    selected_source = st.selectbox(
        "Trend chart data source",
        filter_options,
        key="history_source_filter",
    )
    chart_history = (
        history
        if selected_source == "All sources"
        else [r for r in history if r.get("source") == selected_source]
    )
    st.caption(
        f"Charts plot {len(chart_history)} of {len(history)} run(s) "
        f"(filter: {selected_source}). The table below always lists every "
        "run with its own source."
    )
    cols = st.columns(3)
    for col, key in zip(cols, MEASUREMENT_KEYS):
        with col:
            df = trend_dataframe(chart_history, key)
            if HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["Run"], y=df["Expected"], mode="lines+markers",
                        name="Expected", line=dict(dash="dash", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["Run"], y=df["Observed"], mode="lines+markers",
                        name="Observed", line=dict(width=2),
                    )
                )
                fig.update_layout(
                    title=f"{LABELS[key]} ({UNITS[key]})",
                    height=320, margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=1.12),
                    template="plotly_dark",
                    xaxis_title="Run #",
                )
                st.plotly_chart(fig, theme=None)
            else:  # pragma: no cover - fallback when Plotly is missing
                st.markdown(f"**{LABELS[key]}** ({UNITS[key]})")
                st.line_chart(df.set_index("Run"))

    # ------------------------------------------------------------------ #
    # History table
    # ------------------------------------------------------------------ #
    st.subheader("Previous verification runs")
    st.dataframe(to_dataframe(history), hide_index=True)
    st.caption(
        "Each row shows a concise summary; the full list of reasons for the "
        "most recent run is shown on the **Demonstration** page. History is "
        "temporary (browser session only) and is never sent anywhere."
    )


def render_methodology() -> None:
    st.title("Methodology and limitations")

    st.markdown(
        r"""
        ### Physics-informed integrity verification
        Cyber-physical security monitoring can use the *physics* of a process
        as an independent witness. If a controller claims a certain process
        state, then the physical measurements should look like the expected
        response for that state. When they do not, the claim and the physics
        disagree. This prototype demonstrates that comparison entirely in
        software.

        ### Expected-response calibration
        The dashboard compares observations against an *expected response*:
        one expected value per measurement for the claimed state, plus a
        tolerance. In SIMULATION each scenario carries its own illustrative
        profile; CSV REPLAY and SERIAL runs use the Configuration-page
        values; MANUAL INPUT uses only the reference source you explicitly
        select (typed values, the Configuration page, or an illustrative
        profile) - a missing reference is reported as an error, never
        silently replaced. A real deployment would need experimental
        calibration under defined process conditions, repeated across
        operating points, with measurement uncertainty accounted for.

        ### How the simulator models the process (and what it does not)
        The simulator keeps a bounded, mean-reverting "wander" around an
        illustrative profile, then adds small bounded Gaussian sensor noise.
        Optional effects can be enabled in code: slow sensor drift, a
        measurement delay of N samples, and missing, invalid or stale
        readings. The same seed always reproduces the same sequence, and the
        simulated clock is fixed, so runs are reproducible.

        **What it deliberately does NOT do:** it contains no acid-base
        chemistry. pH cannot be predicted from a universal linear
        dose-to-pH formula; the profiles are invented demonstration numbers.
        No claim of scientific validation is made anywhere in this project.
        The simulator is separated from the detection engine so real
        calibration data can replace the synthetic profiles later without
        rebuilding the dashboard.

        ### Data sources and their honesty labels
        * **SIMULATION** (`SIMULATED`): synthetic values generated on
          demand. Works with no hardware.
        * **MANUAL INPUT** (`MANUAL INPUT`): readings typed by the user.
          They are verified through the same engine as every other source,
          but they are user claims, not automatically verified physical
          measurements. The controller-reported state entered alongside
          them is supporting evidence only - it is never treated as proof
          that the process is normal, and a non-normal claim is never
          automatically treated as an attack.
        * **CSV REPLAY** (`RECORDED`): rows you upload. They are validated
          for structure and plausibility, but their origin is unknown to
          this tool - they are recorded data, not verified measurements.
        * **ESP32 USB SERIAL** (`LIVE HARDWARE`): an optional, read-only
          integration. The app never connects automatically, never sends
          commands, and never operates pumps, relays or dosing equipment.

        The serial integration was developed and unit-tested WITHOUT
        hardware (using a fake serial object), so treat live reads as
        untested until verified on your own bench.

        ### Residual analysis
        For each measurement the engine computes
        `residual = observed - expected` in the measurement's own unit
        (pH dimensionless, conductivity in µS/cm, temperature in °C).
        A residual is a signed difference; the engine flags a measurement
        when the *absolute* residual exceeds its tolerance.

        ### Measurement tolerance
        The tolerance is the maximum acceptable absolute deviation for one
        measurement. The rule is inclusive: `|residual| <= tolerance` passes,
        `|residual| > tolerance` fails. Tolerances must cover ordinary
        measurement noise and sensor accuracy, or the system will produce
        constant alerts.

        ### Data quality, staleness and freshness
        Every sample carries per-measurement quality flags
        (`ok`, `stale`, `missing`, `invalid`) and, in serial mode, an age in
        seconds. A required measurement that is stale, missing or invalid
        can never count as a pass: the engine answers INSUFFICIENT DATA with
        the reason spelled out. The serial mode additionally applies a
        configurable freshness limit, so a frozen or silent sensor is
        reported as stale data rather than trusted.

        ### False positives and false negatives
        * **False positive:** a legitimate disturbance (noise, a sensor
          fault, an unusual but valid operating point) is flagged as an
          integrity anomaly.
        * **False negative:** a manipulation that keeps readings inside the
          tolerances is not flagged.
        No single threshold removes both; the tolerance is a trade-off.

        ### An anomaly is not proof of a cyberattack
        The result **INTEGRITY ANOMALY** means: the numbers disagree beyond
        tolerance, so the claim cannot be trusted as-is and deserves
        investigation. It does not identify an attacker, their motive, their
        method, or even confirm that any malicious activity occurred. This
        prototype makes no claim about who or what caused a discrepancy.

        ### Synthetic demonstration values
        Every default number in this application (pH 8.0, conductivity
        650 µS/cm, temperature 25.5 °C, and the demo tolerances) is an
        invented example chosen to make the demonstration readable. They are
        **not** experimental results, must not be cited as measurements, and
        must not be treated as a calibrated model of any real chemical
        process.

        ### Why this prototype is not industrial-ready
        * No connection to real actuators - by design. The one hardware path
          (serial) is strictly read-only.
        * No calibration study, no uncertainty budget, no safety analysis.
        * History lives only in browser session state (temporary, lost on
          refresh) and there is no authentication, auditing or access
          control.
        * It has not been tested against real attack data and claims neither
          completeness nor any industrial certification or compliance.
        """
    )


# --------------------------------------------------------------------------- #
# Navigation and routing
# --------------------------------------------------------------------------- #

PAGES = {
    "\U0001f3e0 Overview": render_overview,
    "\U0001f52c Demonstration": render_demo,
    "\u2699\ufe0f Configuration": render_config,
    "\U0001f4ca History & Charts": render_history,
    "\U0001f9ed Methodology & Limitations": render_methodology,
}

with st.sidebar:
    st.markdown("## \u2697\ufe0f Chemical Truth Engine 2.0")
    st.caption(
        "Physics-Informed Integrity Verification for Chemical "
        "Cyber-Physical Systems"
    )
    st.divider()
    page_label = st.radio("Navigation", list(PAGES), key="nav_page")
    st.divider()
    st.caption(
        "**EDUCATIONAL SIMULATION** - no real chemical process is controlled. "
        "Values are synthetic unless an ESP32 is explicitly connected in "
        "serial mode."
    )

PAGES[page_label]()

# Rendered after the page so the counter includes runs created on this pass.
with st.sidebar:
    st.caption(f"Runs this session: {len(st.session_state['history'])}")
