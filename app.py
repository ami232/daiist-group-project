"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your deployed app, with one input from
eval/cases.csv (text, or a file path).
"""

import os
import time
from html import escape
from pathlib import Path

import gradio as gr

import env
import inventory
import logbook
import monitor
import pipeline
import store
import trip
import workspace

env.load_env()
logbook.start()

TITLE = "Aisle Run"
UI = Path(__file__).with_name("ui")


def serve(raw_input: str) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    result = pipeline.run(raw_input, products=inventory.catalog())
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


# --- App tab -----------------------------------------------------------------

def on_run(raw_input):
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Give me an input first.")
    record = serve(raw_input)
    outputs = record["outputs"]
    reasons = {
        "unreadable_input": "Upload a readable photo in a supported image format.",
        "missing_key": "Set GEMINI_API_KEY in the server environment and try again.",
        "no_grocery_list": "No grocery list was detected. Try a clearer photo.",
        "items_need_confirmation": "Check the photo or ask a store employee about the unresolved items.",
        "gemini_credentials": "Gemini refused the credentials. Check the key and its project permissions.",
        "gemini_quota": "Gemini is rate-limited or out of quota. Try again after the quota resets.",
        "ai_unavailable": "The AI service is temporarily unavailable. Try again or ask a store employee.",
    }
    status = reasons.get(record["review_reason"], "Your route is ready.")
    agent = outputs["component_b"]
    route = agent["route"]
    matches = []
    for match in agent["matches"]:
        zone = match.get("zone")
        stop = route["stops"].index(zone) + 1 if route and zone in route["stops"] else None
        matches.append([match["index"] + 1, match.get("product") or "", store.NODES[zone]["label"] if zone else "",
                        stop, match["status"]])
    if route:
        status += f" Estimated walk: {route['meters']:.0f} m."
    return (
        outputs["component_a"],
        matches,
        trip.payload(route, agent["matches"]),
        f"**{status}**",
        record["id"],
    )


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


# --- Monitoring tab ----------------------------------------------------------

def summary_markup(rows):
    labels = {
        "requests": "Requests",
        "sent to review": "Sent to review",
        "rated 👍/👎": "Rated requests",
        "share 👎 (of rated)": "Negative feedback",
        "median latency (s)": "Median response (s)",
        "mean cost per request (€)": "Estimated cost per request (€)",
        "total cost (€)": "Total estimated cost (€)",
    }
    tiles = "".join(
        f"<div><dt>{escape(labels.get(row.metric, row.metric))}</dt><dd>{escape(str(row.value))}</dd></div>"
        for row in rows.itertuples(index=False)
    )
    return f'<section class="monitor-panel"><h2>Request summary</h2><dl class="monitor-grid">{tiles}</dl></section>'


def drift_markup(rows, waiting):
    if rows is None or rows.empty:
        body = f"<p>{escape(waiting)}</p>"
    else:
        entries = "".join(
            f"<tr><th scope='row'>{escape(str(row.feature).replace('_', ' ').capitalize())}</th>"
            f"<td>{row.psi:.3f}</td><td>{row.threshold:.3f}</td>"
            f"<td>{'Alert' if row.alert else 'In range'}</td></tr>"
            for row in rows.itertuples(index=False)
        )
        body = (
            "<table class='monitor-table'><thead><tr><th scope='col'>Feature</th>"
            "<th scope='col'>Drift score</th><th scope='col'>Alert threshold</th>"
            f"<th scope='col'>Status</th></tr></thead><tbody>{entries}</tbody></table>"
        )
    return f'<section class="monitor-panel"><h2>Photo brightness drift</h2>{body}</section>'


def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "**Drift baseline pending.** Add the team's reference photos to enable drift comparisons."
        waiting = "The dashboard needs a reference set of normal photos."
    else:
        alert_text = "**Alerts**\n" + "\n".join(f"- {message}" for message in fired) if fired else "**No alerts.**"
        waiting = f"Waiting for {monitor.MIN_WINDOW} requests before comparing recent photos."
    drift = monitor.drift(reference, monitor.live_features(logs)) if reference is not None and len(logs) >= monitor.MIN_WINDOW else None
    return alert_text, summary_markup(monitor.summary(logs)), monitor.quality(logs), drift_markup(drift, waiting)


with gr.Blocks(title=TITLE, delete_cache=(900, 900)) as demo:
    gr.HTML("""
    <div class="aisle-brand"><span class="mark" aria-hidden="true"><i></i></span>
      <div><h1>Aisle Run</h1><p>Find your groceries. Follow your route.</p></div>
    </div>
    """, elem_id="brand-header", apply_default_css=False)
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tabs(selected="shopper", elem_id="app-navigation"):
        with gr.Tab("App", id="shopper"):
            with gr.Tabs(selected="catalog", elem_id="workspace-navigation"):
                with gr.Tab("Browse aisles", id="catalog") as browse_tab:
                    gr.Markdown("## Find a product", elem_classes=["screen-heading"])
                    with gr.Row(elem_id="catalog-tools"):
                        product_search = gr.Textbox(
                            label="Search products", placeholder="Try eggs or pasta", scale=3,
                        )
                        refresh_catalog = gr.Button("Refresh stock", elem_id="refresh-products")
                    aisle_filter = gr.Radio(
                        label="Filter by aisle",
                        choices=[("All aisles", ""), *((aisle["label"], aisle["label"]) for aisle in store.DATA["aisles"])],
                        value="", elem_id="aisle-filter", container=False,
                    )
                    gr.HTML(
                        '<p class="shelf-note">Shelf sides are shown facing the back of the store. Stock counts are estimates.</p>',
                        apply_default_css=False, elem_id="catalog-note",
                    )
                    browse_catalog = gr.HTML(apply_default_css=False, elem_id="catalog-results")
                    filters = [aisle_filter, product_search]
                    demo.load(workspace.browse_markup, filters, browse_catalog, api_name=False)
                    refresh_catalog.click(workspace.browse_markup, filters, browse_catalog, api_name=False)
                    product_search.change(workspace.browse_markup, filters, browse_catalog, api_name=False, trigger_mode="always_last")
                    aisle_filter.change(workspace.browse_markup, filters, browse_catalog, api_name=False)
                    browse_tab.select(workspace.browse_markup, filters, browse_catalog, api_name=False)

                with gr.Tab("Shopping list & route", id="shopping-trip"):
                    request_id = gr.State("")
                    with gr.Accordion("Upload a shopping list", open=False, elem_id="upload-my-list"):
                        with gr.Row():
                            raw_input = gr.Image(type="filepath", label="Shopping-list photo", sources=["upload"],
                                                 format="png", image_mode=None, height=185)
                            with gr.Column():
                                gr.Markdown("Upload a JPG, PNG, or HEIC photo of your list, up to 12 MB. Leave out personal details.")
                                run_button = gr.Button("Find my route", variant="primary", elem_id="find-route")
                    status = gr.Markdown(elem_id="route-feedback")
                    map_view = gr.HTML(
                        value=trip.example_payload(), label="Interactive route",
                        html_template=(UI / "trip.html").read_text(),
                        js_on_load=(UI / "trip.js").read_text(), apply_default_css=False,
                        floor_plan=trip.floor_plan_data_url(),
                    )
                    restore_example = gr.Button("Load example again", size="sm", elem_id="restore-example")
                    with gr.Accordion("Catalog matches and feedback", open=False):
                        out_b = gr.Dataframe(headers=["List line", "Product", "Aisle and side", "Stop", "Status"],
                                             datatype=["number", "str", "str", "number", "str"], interactive=False,
                                             label="Catalog matches")
                        out_a = gr.JSON(label="Vision results")
                        with gr.Row():
                            up = gr.Button("👍 Useful")
                            down = gr.Button("👎 Needs work")
                    request_outputs = [out_a, out_b, map_view, status, request_id]
                    run_button.click(on_run, raw_input, request_outputs, api_name="run")
                    restore_example.click(
                        lambda: (None, [], trip.example_payload(), "", ""), outputs=request_outputs, api_name=False
                    )
                    up.click(lambda rid: on_feedback(rid, +1), request_id, None, api_name=False)
                    down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)

                with gr.Tab("Store staff", id="staff"):
                    staff_session = gr.State(None, time_to_live=workspace.STAFF_SESSION_SECONDS)
                    gr.Markdown("## Store staff\n\nManage products, shelf locations and stock estimates.", elem_classes=["screen-heading"])
                    with gr.Row(elem_id="staff-login") as staff_login_form:
                        staff_code = gr.Textbox(label="Staff access code", type="password", scale=3)
                        sign_in = gr.Button("Sign in", variant="primary")
                    with gr.Row(elem_id="staff-status-row"):
                        staff_status = gr.Markdown()
                        sign_out = gr.Button("Sign out", visible=False, elem_id="staff-sign-out")
                    with gr.Column(visible=False) as staff_tools:
                        with gr.Tabs(selected="products", elem_id="staff-navigation"):
                            with gr.Tab("Add or edit products", id="products"):
                                editor_product = gr.Dropdown(label="Product", choices=[])
                                name = gr.Textbox(label="Product name")
                                planned = gr.Dropdown(label="Planned aisle and side", choices=workspace.aisle_choices())
                                quantity = gr.Number(label="Estimated visible units", value=0, minimum=0, maximum=9999, precision=0)
                                save = gr.Button("Save product", variant="primary")
                            with gr.Tab("Check shelves", id="shelves"):
                                audit_product = gr.Dropdown(label="Product to check", choices=[])
                                expected = gr.Markdown()
                                observed = gr.Dropdown(label="Where you found it", choices=workspace.aisle_choices())
                                counted = gr.Number(label="Estimated visible units", value=0, minimum=0, maximum=9999, precision=0)
                                confirm = gr.Button("Save shelf check", variant="primary")
                        inventory_status = gr.Markdown()
                        inventory_overview = gr.HTML()
                    auth_outputs = [staff_session, staff_tools, staff_status, staff_code, staff_login_form, sign_out]
                    sign_in.click(workspace.staff_login, staff_code, auth_outputs, api_name=False).then(
                        workspace.staff_setup, staff_session, [editor_product, audit_product, inventory_overview], api_name=False
                    )
                    sign_out.click(workspace.staff_logout, outputs=auth_outputs, api_name=False).then(
                        workspace.staff_setup, staff_session, [editor_product, audit_product, inventory_overview], api_name=False
                    )
                    editor_product.change(workspace.edit_fields, [editor_product, staff_session], [name, planned, quantity], api_name=False)
                    audit_product.change(workspace.shelf_fields, [audit_product, staff_session], [expected, observed, counted], api_name=False)
                    staff_outputs = [inventory_status, editor_product, audit_product, inventory_overview]
                    save.click(workspace.save_product, [staff_session, editor_product, name, planned, quantity], staff_outputs, api_name=False)
                    confirm.click(workspace.check_shelf, [staff_session, audit_product, observed, counted], staff_outputs, api_name=False)

        with gr.Tab("Monitoring", id="monitoring"):
            with gr.Row():
                gr.Markdown("## App monitoring", elem_classes=["screen-heading"])
                refresh = gr.Button("Refresh data", elem_id="monitor-refresh")
            alert_box = gr.Markdown(elem_id="monitor-status")
            summary_table = gr.HTML()
            quality_plot = gr.LinePlot(
                x="time", y="value", color="signal", label="Quality over time",
                x_title="Time", y_title="Share of requests", y_axis_format=".0%", y_lim=[0, 1], height=260,
            )
            drift_table = gr.HTML()
            outputs = [alert_box, summary_table, quality_plot, drift_table]
            refresh.click(on_refresh, None, outputs, api_name=False)
            demo.load(on_refresh, None, outputs, api_name=False)


def launch():
    # Render supplies PORT. The localhost override is useful for development.
    return demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", 7860)),
        css_paths=UI / "theme.css",
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["Inter", "system-ui", "sans-serif"]),
        max_file_size="12mb",
    )


if __name__ == "__main__":
    launch()
