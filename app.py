"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your deployed app, with one input from
eval/cases.csv (text, or a file path).

WP7: the visual layer lives in ui/ (HTML views, theme, presets) and
assets/ (CSS design system, logo, favicon). This file only wires events.
The visible form builds the JSON reading; a hidden text box carries it to the
public `/run` endpoint, which still takes a single argument. Every other event
is `api_visibility="private"`, so it is not published as an API endpoint.
"""

import json
import os
import time
import traceback
from pathlib import Path

import gradio as gr

import env
import logbook
import monitor
import pipeline
from maintai import agent, bandit, component_a
from maintai.schemas import MachineInput
from ui import components as ui
from ui import dashboard
from ui import presets
from ui.theme import build_theme

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "MaintAI – Intelligent Maintenance Advisor"
DESCRIPTION = "For maintenance managers: predicts the failure risk of a machine process and drafts a work order to approve. Plant data and systems are simulated."
ASSETS = Path(__file__).resolve().parent / "assets"
STUBS = ui.stub_parts({"component_a": component_a.MODEL_VERSION, "component_b": agent.PROMPT_VERSION,
                       "bandit": bandit.POLICY_VERSION})


def serve(raw_input: str) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    try:  # WP6: the policy is rebuilt from the logs (Postgres on Render) on every request
        logs, policy_from = logbook.load(), "logs"
    except Exception:  # logs unreachable: choose from the prior alone rather than fail the request
        traceback.print_exc()
        logs, policy_from = None, "prior_only"
    decision = bandit.decide(logs)
    result = pipeline.run(raw_input, arm=decision.arm)
    extra = {"source": os.environ.get("MAINTAI_SOURCE", "real"), "arm_probs": decision.arm_probs,
             "bandit_explore": decision.explore, "bandit_greedy_arm": decision.greedy_arm,
             "bandit_n_feedback": decision.n_feedback, "bandit_policy_from": policy_from}
    return logbook.log_request(result | extra | {"latency_s": round(time.perf_counter() - started, 3)})


# --- App tab -----------------------------------------------------------------

def _limits(field: str) -> tuple[float | None, float | None]:
    """Plausibility limits declared in schemas.MachineInput, so the form never repeats them by hand."""
    low = high = None
    for rule in MachineInput.model_fields[field].metadata:
        low = getattr(rule, "ge", getattr(rule, "gt", low))
        high = getattr(rule, "le", getattr(rule, "lt", high))
    return low, high


def _range_info(field: str) -> str:
    low, high = _limits(field)
    return f"{low:g}–{high:g}"


def build_request(machine_type, air, process, rpm, torque, wear):
    """Form -> the JSON reading for /run. Only checks that every field is filled in:
    range checks belong to the pipeline, which refuses out-of-scope readings and logs them."""
    values = dict(zip(presets.FIELDS, (machine_type, air, process, rpm, torque, wear)))
    if missing := [ui.FIELD_LABELS[k] for k, v in values.items() if v is None or v == ""]:
        raise gr.Error(f"Complete every field before running the analysis: {', '.join(missing)}.", title="Missing values", duration=8)
    return json.dumps(values), ui.loading_state(), gr.update(visible=False), ""


def on_run(raw_input):
    """The public /run endpoint (one text argument, as check.py sends it)."""
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Give me an input first.")
    try:
        record = serve(raw_input)
    except Exception as error:  # shown as the error state; the traceback goes to the server log
        traceback.print_exc()
        raise gr.Error(f"The analysis service failed ({type(error).__name__}).", title="Analysis failed") from error
    if record.get("status") == "out_of_scope":
        return ui.out_of_scope(record), record["id"], "", gr.update(visible=False), ""
    return ui.results(record), record["id"], record.get("arm", ""), gr.update(visible=True), ""


def on_feedback(request_id, arm, value):
    if not request_id:
        gr.Warning("Run an analysis first.")
        return ""
    logbook.log_feedback(request_id, value)
    return ui.feedback_thanks(value, arm)


# --- Monitoring tab ----------------------------------------------------------

def on_refresh():
    """Calculations stay in monitor.py (WP5); ui/dashboard.py only lays them out."""
    logs = logbook.load()
    updated = f'<p class="mai-updated">{ui._e("Updated " + time.strftime("%H:%M:%S"))}</p>'
    return dashboard.render(logs, monitor.load_reference(), STORAGE), updated


# --- Layout ----------------------------------------------------------------------

PRIVATE = {"api_visibility": "private"}
STORAGE = "PostgreSQL (Render)" if logbook.database_url() else "local files"

with gr.Blocks(title=TITLE, fill_width=True) as demo:
    gr.HTML(ui.header(STUBS), elem_classes=["mai-html", "mai-header-wrap"])

    with gr.Column(elem_classes="mai-shell"):
        gr.HTML(ui.missing_env_banner(env.missing()) + ui.dev_banner(STUBS), elem_classes="mai-html")

        with gr.Tabs(elem_classes="mai-nav"):
            with gr.Tab("App", id="app"):
                gr.HTML(ui.page_title("Machine risk analysis", DESCRIPTION, "Maintenance advisor"), elem_classes="mai-html")
                request_id = gr.State("")
                arm_state = gr.State("")  # bandit strategy, revealed only after feedback

                with gr.Row(elem_classes="mai-layout", equal_height=False):
                    with gr.Column(scale=4, min_width=300, elem_classes=["mai-panel", "mai-form"]):
                        gr.HTML(ui.section_title("Machine reading", "sliders", hint="AI4I 2020 units"), elem_classes="mai-html")
                        sample = gr.Dropdown(list(presets.PRESETS), value=None, label="Load a sample reading",
                                             info="Hand-written scenarios, not plant data", elem_classes="mai-sample")
                        start = presets.PRESETS[presets.DEFAULT]
                        machine_type = gr.Radio([("L · Low", "L"), ("M · Medium", "M"), ("H · High", "H")],
                                                value=start["type"], label="Machine type (quality variant)", elem_classes="mai-type")
                        with gr.Row(elem_classes="mai-fields"):
                            air = gr.Number(start["air_temperature_k"], label="Air temperature · K", step=0.1,
                                            info=_range_info("air_temperature_k"), min_width=120)
                            process = gr.Number(start["process_temperature_k"], label="Process temp. · K", step=0.1,
                                                info=_range_info("process_temperature_k"), min_width=120)
                        with gr.Row(elem_classes="mai-fields"):
                            rpm = gr.Number(start["rotational_speed_rpm"], label="Speed · rpm", step=10,
                                            info="Above 0, up to 5000", min_width=120)
                            torque = gr.Number(start["torque_nm"], label="Torque · Nm", step=0.5,
                                               info=_range_info("torque_nm"), min_width=120)
                        with gr.Row(elem_classes="mai-fields"):
                            wear = gr.Number(start["tool_wear_min"], label="Tool wear · min", step=1,
                                             info=_range_info("tool_wear_min"), min_width=120)
                            gr.HTML('<p class="mai-form__note">Readings outside these ranges are refused before any model runs.</p>',
                                    elem_classes="mai-html", min_width=120)
                        form = [machine_type, air, process, rpm, torque, wear]
                        run_button = gr.Button("Analyse machine", variant="primary", elem_classes="mai-run")
                        # Carries the JSON to /run (one text argument, as check.py expects). Not shown.
                        raw_input = gr.Textbox(label="Machine reading (JSON)", visible="hidden")

                    with gr.Column(scale=8, min_width=320):
                        results = gr.HTML(ui.empty_state(), elem_classes="mai-html")
                        with gr.Row(visible=False, elem_classes=["mai-panel", "mai-feedback"]) as feedback_row:
                            gr.HTML('<p><strong>Was this work order useful?</strong><br>'
                                    '<span class="mai-subtle">Your rating is logged and helps MaintAI improve how it writes orders.</span></p>',
                                    elem_classes=["mai-html", "mai-feedback__q"])
                            up = gr.Button("Useful", variant="secondary", elem_classes="mai-feedback__btn",
                                           icon=str(ASSETS / "icons" / "thumb-up.svg"))
                            down = gr.Button("Not useful", variant="secondary", elem_classes="mai-feedback__btn",
                                             icon=str(ASSETS / "icons" / "thumb-down.svg"))
                            thanks = gr.HTML("", elem_classes=["mai-html", "mai-feedback__result"])

                sample.input(lambda name: presets.values(name) if name else [gr.skip()] * len(form), sample, form,
                             show_progress="hidden", **PRIVATE)

                run_outputs = [results, request_id, arm_state, feedback_row, thanks]
                (run_button
                 .click(build_request, form, [raw_input, results, feedback_row, thanks], show_progress="hidden", **PRIVATE)
                 .success(on_run, raw_input, run_outputs, api_name="run", show_progress="hidden")
                 .failure(lambda: ui.error_state("The service did not return a result."), None, results, **PRIVATE))
                up.click(lambda rid, arm: on_feedback(rid, arm, +1), [request_id, arm_state], thanks, **PRIVATE)
                down.click(lambda rid, arm: on_feedback(rid, arm, -1), [request_id, arm_state], thanks, **PRIVATE)

            with gr.Tab("Monitoring", id="monitoring", elem_classes="mai-monitor"):
                with gr.Row(equal_height=False, elem_classes="mai-pagehead-row"):
                    gr.HTML(ui.page_title("Operations monitoring", "Usage, quality, drift and cost, computed from the request logs.",
                                          "Monitoring"), elem_classes="mai-html")
                    with gr.Column(scale=0, min_width=150, elem_classes="mai-refresh"):
                        updated = gr.HTML(elem_classes="mai-html")
                        refresh = gr.Button("Refresh", variant="secondary", size="sm",
                                            icon=str(ASSETS / "icons" / "refresh.svg"))
                dashboard_html = gr.HTML(elem_classes="mai-html")
                refresh.click(on_refresh, None, [dashboard_html, updated], **PRIVATE)
                demo.load(on_refresh, None, [dashboard_html, updated], **PRIVATE)

        gr.HTML(ui.footer(STORAGE), elem_classes="mai-html")


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                theme=build_theme(), css_paths=[ASSETS / "styles.css"], favicon_path=str(ASSETS / "favicon.svg"))
