"""Tender Desk's Gradio interface and assignment-compatible /run endpoint."""

import csv
import os
import time
from html import escape
from pathlib import Path

import gradio as gr

import env
import logbook
import monitor
import pipeline

env.load_env()
logbook.start()

TITLE = "Tender Desk"
LABELS = {"bid": "Shortlist for bid", "no_bid": "Do not bid", "review": "Needs review"}
CSS = Path(__file__).with_name("tender.css").read_text(encoding="utf-8")
HEADER = """
<div class="td-brand">
    <span class="td-mark" aria-hidden="true"></span>
    <span>Tender Desk</span>
</div>
<header class="td-hero">
    <div class="td-hero-copy">
        <p class="td-scope">Spanish public construction tenders</p>
        <h1>Should we bid?</h1>
        <p>Paste a public tender to see its key details, estimated competition, and whether it needs a closer look.</p>
    </div>
    <div class="td-blueprint" aria-hidden="true"><div class="td-plan"></div></div>
</header>
"""
EMPTY_RESULT = (
    "Your recommendation will appear here after you select **Analyse tender**. "
    "You’ll also see the contract details and an estimate of how many offers it may receive."
)
FIT_CHOICES = [
    ("Eligibility not checked", pipeline.DEFAULT_FIT),
    ("Company meets all mandatory requirements", pipeline.FIT_OPTIONS[1]),
    ("Company fails a mandatory requirement", pipeline.FIT_OPTIONS[2]),
]
CASE_PLAN = Path(__file__).with_name("eval") / "case-plan.csv"
with CASE_PLAN.open(newline="", encoding="utf-8") as case_file:
    TEAM_CASES = {row["id"]: row["input"] for row in csv.DictReader(case_file)}


def prepare_notice(notice: str) -> tuple:
    """Fill the form and clear any result left by the previous notice."""
    return notice, pipeline.DEFAULT_FIT, EMPTY_RESULT, "", None, ""



def serve(raw_input: str, fit: str = pipeline.DEFAULT_FIT) -> dict:
    started = time.perf_counter()
    result = pipeline.run(raw_input, fit)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


def on_run(raw_input, fit):
    if not (raw_input or "").strip():
        raise gr.Error("Paste a construction tender or load the TED example.")
    record = serve(raw_input, fit)
    outputs = record["outputs"]
    facts = outputs["component_a"]
    result = outputs["component_b"]
    decision = result["decision"]
    guidance = f"### {LABELS[decision]}\n{result['reason']}\n"
    estimate = result["estimated_offers"]
    if estimate is not None:
        radius = result["typical_error"]
        guidance += (
            f"\n- **{estimate}** Estimated offers\n"
            f"- **{max(1, estimate - radius)}–{estimate + radius}** Historical error range"
        )
    budget = facts.get("budget_eur")
    duration = facts.get("duration_months")
    award_basis = {"price_quality": "Multiple criteria", "price": "Lowest price"}.get(facts.get("award_basis"), "Unknown")
    summary = "#### Notice details\nNo notice fields available."
    if facts:
        summary = (
            "#### Notice details\n"
            f"- Budget excl. VAT **{'€' + format(budget, ',.2f') if budget is not None else 'Unknown'}**\n"
            f"- Duration **{format(duration, 'g') + ' months' if duration is not None else 'Unknown'}**\n"
            f"- Construction code (CPV) **{facts.get('cpv') or 'Unknown'}**\n"
            f"- Award criteria **{award_basis}**"
        )
    return guidance, summary, outputs, record["id"]


def feedback(request_id, value):
    if not request_id:
        gr.Warning("Analyse a tender first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


def refresh_monitor():
    logs = logbook.load()
    reference = monitor.load_reference()
    if reference is None:
        message = "The team’s reference set is pending. Review rate, latency and cost are recorded below."
    elif len(logs) < monitor.MIN_WINDOW or "f_outside_construction" not in logs:
        message = f"Sector drift needs {monitor.MIN_WINDOW} recent notices with an explicit primary CPV."
    else:
        alerts = monitor.alerts(logs, reference)
        has_sector_window = monitor.live_features(logs)["outside_construction"].count() >= monitor.MIN_WINDOW
        if alerts:
            message = "\n".join(f"- {alert}" for alert in alerts)
        elif has_sector_window:
            message = "No alerts in the current window."
        else:
            message = f"Sector drift needs {monitor.MIN_WINDOW} recent notices with an explicit primary CPV."
    drift = None
    if reference is not None and len(logs) >= monitor.MIN_WINDOW and "f_outside_construction" in logs:
        drift = monitor.drift(reference, monitor.live_features(logs))
    labels = {
        "requests": "Requests", "sent to review": "Sent to review",
        "rated 👍/👎": "Rated requests", "share 👎 (of rated)": "Negative feedback",
        "median latency (s)": "Median latency (s)",
        "mean cost per request (€)": "Cost per request (€)", "total cost (€)": "Total cost (€)",
    }
    metrics = "".join(
        f"<div><dt>{escape(labels[row.metric])}</dt><dd>{escape(str(row.value))}</dd></div>"
        for row in monitor.summary(logs).itertuples()
    )
    summary = f'<section aria-label="Request summary"><h3>Request summary</h3><dl class="td-metrics">{metrics}</dl></section>'
    return message, summary, monitor.quality(logs), gr.update(value=drift, visible=drift is not None)


with gr.Blocks(title=TITLE) as demo:
    gr.HTML(HEADER, elem_id="tender-header", js_on_load=None, padding=False)
    with gr.Tabs(elem_id="tender-tabs"):
        with gr.Tab("App"):
            request_id = gr.State("")
            with gr.Row(elem_id="tender-workspace"):
                with gr.Column(scale=6, min_width=0, elem_classes=["td-panel", "td-form"]):
                    with gr.Row(elem_classes="td-panel-heading"):
                        gr.Markdown("## 1. Add a tender", elem_classes="td-section-title")
                        sample = gr.Button("Load TED example", size="sm", scale=0, min_width=0,
                                           elem_id="tender-example")
                    gr.Markdown(
                        "Paste the text from a Spanish public construction notice, or load the example to try it.",
                        elem_classes="td-help",
                    )
                    with gr.Accordion("Load an assigned test case", open=False, elem_id="tender-team"):
                        gr.Markdown(
                            "Write down your expected answers before running a case. "
                            "[Open the case plan](https://github.com/Jnavarro-web/daiist-group-project/blob/codex/tender-tool/eval/case-plan.csv) "
                            "for the original notice and award links.",
                            elem_classes="td-help",
                        )
                        case_id = gr.Dropdown(
                            list(TEAM_CASES), value=None, num_choices_shown=7,
                            label="Choose your case ID (c01–c30)", elem_id="tender-case",
                        )
                    raw_input = gr.Textbox(
                        label="Tender notice text", show_label=False,
                        placeholder="Paste the tender notice text in Spanish or English…",
                        lines=6, max_lines=16, elem_id="tender-input",
                    )
                    gr.Markdown(
                        "Tender Desk automatically removes common contact details before analysis. "
                        "You can preview the cleaned text and remove anything it misses.",
                        elem_classes="td-help",
                    )
                    clean = gr.Button("Remove contact details", size="sm", elem_id="tender-clean")
                    run_button = gr.Button("Analyse tender", variant="primary", elem_id="tender-analyse")
                    with gr.Accordion("Company eligibility (optional)", open=False, elem_id="tender-eligibility"):
                        gr.Markdown(
                            "Tender Desk reads the notice and estimates competition. If you’ve already "
                            "checked that your company has the required experience or certifications, "
                            "record the outcome here. For a quick test, select **Analyse tender** directly.",
                            elem_classes="td-help",
                        )
                        fit = gr.Radio(
                            FIT_CHOICES, value=pipeline.DEFAULT_FIT, container=False,
                            show_label=False, label="Company eligibility", elem_id="tender-fit",
                        )
                with gr.Column(scale=5, min_width=0, elem_classes=["td-panel", "td-result"]):
                    gr.Markdown("## 2. Read the result", elem_classes="td-section-title")
                    guidance = gr.Markdown(EMPTY_RESULT, elem_id="tender-guidance")
                    with gr.Column(elem_id="tender-expanded"):
                        summary = gr.Markdown(elem_id="tender-facts")
                        with gr.Accordion("Technical details", open=False):
                            details = gr.JSON(show_label=False)
                        with gr.Row(elem_classes="td-feedback"):
                            up = gr.Button("Useful", size="sm")
                            down = gr.Button("Needs improvement", size="sm")
            form = [raw_input, fit, guidance, summary, details, request_id]
            sample.click(
                lambda: prepare_notice((pipeline.DATA / "example.txt").read_text(encoding="utf-8")),
                None, form, api_name=False,
            )
            clean.click(pipeline.sanitize, raw_input, raw_input, api_name=False)
            case_id.input(
                lambda selected: prepare_notice(TEAM_CASES.get(selected, "")),
                case_id, form, api_name=False,
            )
            run_button.click(
                on_run, [raw_input, fit], [guidance, summary, details, request_id],
                api_name="run", scroll_to_output=True,
            )
            up.click(lambda rid: feedback(rid, 1), request_id, None, api_name=False)
            down.click(lambda rid: feedback(rid, -1), request_id, None, api_name=False)

        with gr.Tab("Monitoring"):
            with gr.Column(elem_classes="td-panel"):
                with gr.Row(elem_id="tender-monitor-heading"):
                    gr.Markdown("## App monitoring", elem_classes="td-section-title")
                    refresh = gr.Button("Refresh metrics", scale=0, min_width=0, elem_id="tender-refresh")
                alert = gr.Markdown(elem_id="tender-monitor-alert")
                summary_cards = gr.HTML(js_on_load=None, elem_id="tender-monitor-summary")
                quality_plot = gr.LinePlot(
                    x="time", y="value", color="signal", label="Review and feedback over time",
                    y_axis_format=".0%", y_lim=[0, 1], x_title="Time", y_title="Share of requests",
                    height=310,
                )
                drift_table = gr.Dataframe(label="Changes from the reference set", interactive=False, visible=False)
                outputs = [alert, summary_cards, quality_plot, drift_table]
                refresh.click(refresh_monitor, None, outputs, api_name=False)
                demo.load(refresh_monitor, None, outputs, api_name=False)


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(
            primary_hue="teal", neutral_hue="slate",
            font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
        ),
        css=CSS, footer_links=[],
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")), show_error=False,
    )
