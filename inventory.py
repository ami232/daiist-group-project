"""Store catalog edits and shelf observations, with SQLite locally and Postgres on Render."""

import json
import os
import re
import sqlite3
import unicodedata
from contextlib import closing, contextmanager
from pathlib import Path

import store

LOCAL_DB = Path(__file__).with_name("logs") / "inventory.sqlite3"
AISLES = {zone: store.NODES[zone]["label"] for aisle in store.DATA["aisles"] for zone in aisle["sections"]}
SCHEMA = "CREATE TABLE IF NOT EXISTS aisle_run_inventory (id TEXT PRIMARY KEY, record TEXT NOT NULL)"


@contextmanager
def database(*, write=False):
    url = os.environ.get("DATABASE_URL", "").strip()
    if url and not os.environ.get("LOG_DIR"):
        import psycopg

        with psycopg.connect(url, connect_timeout=8) as connection:
            connection.execute(SCHEMA)
            if write:
                connection.execute("SELECT pg_advisory_xact_lock(718203127)")
            yield connection, "%s"
        return
    log_dir = os.environ.get("LOG_DIR")
    local_db = Path(log_dir) / LOCAL_DB.name if log_dir else LOCAL_DB
    local_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(local_db, timeout=8)) as connection, connection:
        connection.execute(SCHEMA)
        if write:
            connection.execute("BEGIN IMMEDIATE")
        yield connection, "?"


def _products(connection):
    products = {key: dict(item) for key, item in store.PRODUCTS.items()}
    for product_id, raw in connection.execute("SELECT id, record FROM aisle_run_inventory").fetchall():
        products[product_id] = json.loads(raw)
    return products


def catalog():
    with database() as (connection, _):
        return _products(connection)


def location_issue(product):
    return product.get("observed_zone", product["zone"]) != product["zone"]


def _write(connection, marker, product):
    connection.execute(
        f"INSERT INTO aisle_run_inventory (id, record) VALUES ({marker}, {marker}) "
        "ON CONFLICT (id) DO UPDATE SET record = excluded.record",
        (product["id"], json.dumps(product, ensure_ascii=False)),
    )


def _count(value):
    if not isinstance(value, (int, float)) or not 0 <= value <= 9999 or int(value) != value:
        raise ValueError("Enter an estimated whole number from 0 to 9999.")
    return int(value)


def save_product(selected, name, zone, count):
    name = " ".join((name or "").split())
    if not name or len(name) > 75:
        raise ValueError("Enter a product name up to 75 characters.")
    if zone not in AISLES:
        raise ValueError("Choose the aisle and shelf side.")
    count = _count(count)
    slug_base = unicodedata.normalize("NFKD", name.casefold()).encode("ascii", "ignore").decode()
    product_id = selected or re.sub(r"[^a-z0-9]+", "-", slug_base).strip("-")
    if not product_id:
        raise ValueError("Choose a product name with at least one letter or number.")
    with database(write=True) as (connection, marker):
        products = _products(connection)
        if selected and selected not in products:
            raise ValueError("Choose an existing product or Add new product.")
        if not selected and product_id in products:
            raise ValueError("This product already exists. Select it to make changes.")
        product = products.get(product_id, {"id": product_id, "aliases": []})
        updated = product | {"name": name, "zone": zone, "count": count}
        _write(connection, marker, updated)
        return updated


def check_shelf(product_id, observed_zone, count):
    if observed_zone not in AISLES:
        raise ValueError("Choose the shelf where you found the product.")
    count = _count(count)
    with database(write=True) as (connection, marker):
        product = _products(connection).get(product_id)
        if product is None:
            raise ValueError("Choose a product to check.")
        updated = product | {"observed_zone": observed_zone, "count": count}
        _write(connection, marker, updated)
        return updated
