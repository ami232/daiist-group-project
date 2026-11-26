"""Customer catalog and the gated staff workspace."""

import hmac
import os
import time
from html import escape

import gradio as gr

import inventory
import store

STAFF_CODE = "AISLE_RUN_STAFF_CODE"
STAFF_SESSION_SECONDS = 4 * 60 * 60


def aisle_choices():
    return [(label, key) for key, label in inventory.AISLES.items()]


def product_choices():
    return [(product["name"], product["id"]) for product in sorted(
        inventory.catalog().values(), key=lambda entry: entry["name"].casefold()
    )]


def browse_markup(aisle=None, search=None):
    query = (search or "").strip().casefold()
    grouped = {}
    for product in sorted(inventory.catalog().values(), key=lambda entry: entry["name"].casefold()):
        if query and not any(query in label.casefold() for label in [product["name"], *product["aliases"]]):
            continue
        issue = inventory.location_issue(product)
        stock = "Location under review" if issue else (
            f'~{product["count"]} left' if product["count"] else "Out of stock"
        )
        stock_class = "stock-attention" if issue or not product["count"] else "stock-present"
        grouped.setdefault(product["zone"], []).append(
            f'<li><strong>{escape(product["name"])}</strong><span class="{stock_class}">{stock}</span></li>'
        )

    aisles = []
    for entry in store.DATA["aisles"]:
        label = entry["label"]
        if aisle and aisle != label:
            continue
        sections = entry["sections"]
        if not any(zone in grouped for zone in sections):
            continue
        shelves = []
        for zone in sections:
            side = inventory.AISLES[zone].split(" · ")[-1]
            items = grouped.get(zone)
            content = f'<ul>{"".join(items)}</ul>' if items else '<p class="shelf-empty">No matching products</p>'
            shelves.append(f'<div class="shelf"><h4>{escape(side)} shelf</h4>{content}</div>')
        aisles.append(
            f'<section class="browse-aisle"><h3>{escape(label)}</h3>'
            f'<div class="shelf-pair">{"".join(shelves)}</div></section>'
        )
    body = "".join(aisles) or '<p class="catalog-empty">No products found. Try another search or aisle.</p>'
    return f'<div class="browse-grid" aria-live="polite">{body}</div>'


def staff_is_signed_in(session):
    if not isinstance(session, tuple) or len(session) != 2:
        return False
    expires, signature = session
    code = os.getenv(STAFF_CODE, "").strip()
    return (
        bool(code)
        and isinstance(expires, (int, float))
        and isinstance(signature, str)
        and expires > time.time()
        and hmac.compare_digest(signature, _sign(code, expires))
    )


def _sign(code, expires):
    return hmac.digest(code.encode(), str(expires).encode(), "sha256").hex()


def staff_login(code):
    configured = os.getenv(STAFF_CODE, "").strip()
    if not configured:
        return None, gr.update(visible=False), "Staff access is not configured.", "", gr.update(visible=True), gr.update(visible=False)
    if not hmac.compare_digest((code or "").encode(), configured.encode()):
        return None, gr.update(visible=False), "That access code didn't match.", "", gr.update(visible=True), gr.update(visible=False)
    expires = time.time() + STAFF_SESSION_SECONDS
    return (expires, _sign(configured, expires)), gr.update(visible=True), "Signed in as store staff.", "", gr.update(visible=False), gr.update(visible=True)


def staff_logout():
    return None, gr.update(visible=False), "Signed out.", "", gr.update(visible=True), gr.update(visible=False)


def require_staff(session):
    if not staff_is_signed_in(session):
        raise gr.Error("Sign in to the staff workspace first.")


def edit_fields(selected, session):
    require_staff(session)
    product = inventory.catalog().get(selected, {})
    return product.get("name", ""), product.get("zone"), product.get("count", 0)


def shelf_fields(selected, session):
    require_staff(session)
    product = inventory.catalog().get(selected)
    if not product:
        return "", None, 0
    return f"Planned shelf · **{inventory.AISLES[product['zone']]}**", product.get("observed_zone", product["zone"]), product["count"]


def staff_markup(session):
    require_staff(session)
    items = []
    for product in sorted(inventory.catalog().values(), key=lambda p: p["name"].casefold()):
        expected = inventory.AISLES[product["zone"]]
        observed = inventory.AISLES[product.get("observed_zone", product["zone"])]
        if inventory.location_issue(product):
            status = f"Misplaced · Found at {observed}"
        else:
            status = "Shelf matches" if "observed_zone" in product else "Awaiting shelf check"
        tone = "stock-attention" if inventory.location_issue(product) or not product["count"] else "stock-present"
        items.append(
            f'<li><strong>{escape(product["name"])}</strong><span>{escape(expected)}</span>'
            f'<span class="{tone}">{escape(status)} · {product["count"]} visible</span></li>'
        )
    return f'<section class="staff-overview"><h3>Shelf status</h3><ul>{"".join(items)}</ul></section>'


def save_product(session, selected, name, zone, quantity):
    require_staff(session)
    try:
        product = inventory.save_product(selected, name, zone, quantity)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return changes(session, product, "Saved the product and its planned shelf.")


def check_shelf(session, product_id, zone, quantity):
    require_staff(session)
    try:
        product = inventory.check_shelf(product_id, zone, quantity)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    message = "Shelf check saved. Marked as misplaced until staff confirm the correct shelf." if inventory.location_issue(product) else "Shelf check saved. The shelf matches the catalog."
    return changes(session, product, message)


def changes(session, product, message):
    choices = product_choices()
    return (
        message,
        gr.update(choices=[("Add new product", ""), *choices], value=product["id"]),
        gr.update(choices=choices, value=product["id"]),
        staff_markup(session),
    )


def staff_setup(session):
    if not staff_is_signed_in(session):
        return gr.skip(), gr.skip(), ""
    choices = product_choices()
    return gr.update(choices=[("Add new product", ""), *choices], value=""), gr.update(choices=choices, value=None), staff_markup(session)
