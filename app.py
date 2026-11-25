"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your deployed app, with one input from
eval/cases.csv (text, or a file path).
"""

import json
import os
import time

import gradio as gr

import env
import logbook
import monitor
import pipeline
from maintai import agent, bandit, component_a
from maintai.inputs import EXAMPLE_INPUT

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "MaintAI – Intelligent Maintenance Advisor"
DESCRIPTION = "For maintenance managers: predicts the failure risk of a machine process and drafts a work order to approve. Plant data and systems are simulated."


def serve(raw_input: str) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    arm, arm_probs = bandit.choose_arm()  # WP6: pass logbook.load() once the bandit learns
    result = pipeline.run(raw_input, arm=arm)
    extra = {"source": os.environ.get("MAINTAI_SOURCE", "real"), "arm_probs": arm_probs}
    return logbook.log_request(result | extra | {"latency_s": round(time.perf_counter() - started, 3)})


def _work_order_text(record: dict) -> str:
    """Plain-text view of the work order (WP7 replaces it with the designed card)."""
    b = record["outputs"]["component_b"]
    if record.get("status") == "out_of_scope":
        return f"Cannot help with this input ({b['reason']}). {b.get('detail', '')}"
    order = b["work_order"]
    draft = order["draft"]
    lines = [f"{draft['priority']} · {draft['action']} · mode {draft['failure_mode_addressed']}", draft["justification"]]
    lines += [f"- {step}" for step in draft["steps"]]
    if order.get("cost"):
        lines.append(f"Preventive €{order['cost']['preventive_eur']:.0f} vs breakdown €{order['cost']['breakdown_eur']:.0f}")
    return "\n".join(lines)


# --- App tab -----------------------------------------------------------------

def on_run(raw_input):
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Give me an input first.")
    record = serve(raw_input)
    if record.get("status") == "out_of_scope":
        status = "⛔ **Out of scope:** no model was called."
    elif record["review"]:
        status = "⚠️ **Sent to a person for review.**"
    else:
        status = "✅ Handled automatically."
    if record.get("degraded"):
        status += f"  _Degraded mode: {record.get('degraded_reason', '')}._"
    return (
        record["outputs"]["component_a"],
        _work_order_text(record),
        f"{status}  Latency {record['latency_s']} s, cost €{record['cost_eur']:.4f}.",
        record["id"],
    )


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


# --- Monitoring tab ----------------------------------------------------------

def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "No reference set yet: run `uv run python main.py reference` and commit `monitoring/`."
    elif fired:
        alert_text = "### 🚨 Alerts\n" + "\n".join(f"- {message}" for message in fired)
    else:
        alert_text = "### ✅ No alerts"
    if reference is not None and len(logs) >= monitor.MIN_WINDOW:
        drift_table = monitor.drift(reference, monitor.live_features(logs))
    else:
        drift_table = monitor.drift(reference, reference.head(0)) if reference is not None else None
    return alert_text, monitor.summary(logs), monitor.quality(logs), drift_table


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")
    if stubs := [v for v in (component_a.MODEL_VERSION, agent.PROMPT_VERSION, bandit.POLICY_VERSION) if v.startswith("stub")]:
        gr.Markdown(f"🚧 **Development version.** These parts are provisional stubs, so their outputs are "
                    f"placeholders, not real predictions: {', '.join(stubs)}.")

    with gr.Tab("App"):
        request_id = gr.State("")
        # WP7: replace with a form that builds this JSON; keep one text input for the /run endpoint.
        raw_input = gr.Textbox(label="Machine reading (JSON)", lines=4, value=json.dumps(EXAMPLE_INPUT))
        run_button = gr.Button("Analyse machine", variant="primary")
        with gr.Row():
            out_a = gr.JSON(label="Component A · failure risk")
            out_b = gr.Textbox(label="Component B · work order", lines=8)
        status = gr.Markdown()
        with gr.Row():
            up = gr.Button("👍")
            down = gr.Button("👎")
        run_button.click(on_run, raw_input, [out_a, out_b, status, request_id], api_name="run")
        up.click(lambda rid: on_feedback(rid, +1), request_id, None, api_name=False)
        down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)

    with gr.Tab("Monitoring"):
        refresh = gr.Button("Refresh")
        alert_box = gr.Markdown()
        summary_table = gr.Dataframe(label="Summary", interactive=False)
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", label="Quality over time", height=300)
        drift_table = gr.Dataframe(label=f"Drift: PSI of the last {monitor.WINDOW} requests against the reference set", interactive=False)
        gr.Markdown(f"<small>{LOG_TARGET}</small>")
        outputs = [alert_box, summary_table, quality_plot, drift_table]
        refresh.click(on_refresh, None, outputs, api_name=False)
        demo.load(on_refresh, None, outputs, api_name=False)


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
