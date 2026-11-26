"""Aisle Run's sample store and its deterministic walking graph."""

import json
from collections import defaultdict
from heapq import heappop, heappush
from itertools import pairwise, permutations
from math import hypot
from pathlib import Path

from PIL import Image, ImageDraw

DATA = json.loads(Path(__file__).with_name("store.json").read_text())
PRODUCTS = {item["id"]: item for item in DATA["catalog"]}
NODES = {node["id"]: node for node in DATA["nodes"]}
GRAPH = defaultdict(list)
for a, b in DATA["paths"]:
    length = hypot(NODES[a]["x"] - NODES[b]["x"], NODES[a]["y"] - NODES[b]["y"])
    GRAPH[a].append((b, length))
    GRAPH[b].append((a, length))


def shortest_paths(start):
    distance, previous, pending = {start: 0}, {}, [(0, start)]
    while pending:
        length, node = heappop(pending)
        if length > distance[node]:
            continue
        for neighbor, step in GRAPH[node]:
            candidate = length + step
            if candidate < distance.get(neighbor, float("inf")):
                distance[neighbor], previous[neighbor] = candidate, node
                heappush(pending, (candidate, neighbor))
    return distance, previous


def walking_path(previous, start, end):
    leg = [end]
    while leg[-1] != start:
        leg.append(previous[leg[-1]])
    return list(reversed(leg))


def plan(zones):
    distances = {node: shortest_paths(node) for node in {"entrance", *zones}}

    def length(stops):
        return sum(distances[a][0][b] for a, b in pairwise(["entrance", *stops, "checkout"]))

    ordered = min(permutations(dict.fromkeys(zones)), key=length)
    trace = ["entrance"]
    for start, end in pairwise(["entrance", *ordered, "checkout"]):
        trace.extend(walking_path(distances[start][1], start, end)[1:])
    scale = DATA["metersPerUnit"]
    return {
        "stops": list(ordered),
        "trace": trace,
        "meters": round(length(ordered) * scale, 2),
        "list_order_meters": round(length(zones) * scale, 2),
    }


def render_floor_plan():
    image = Image.new("RGB", (800, 500), "#0d1929")
    draw = ImageDraw.Draw(image)

    def point(node):
        return NODES[node]["x"], NODES[node]["y"]

    draw.rectangle((36, 34, 764, 466), outline="#2f4a60", width=2)
    for a, b in DATA["paths"]:
        x, y = point(a)
        bx, by = point(b)
        dx, dy = bx - x, by - y
        length = hypot(dx, dy)
        for d in range(0, int(length), 14):
            end = min(d + 7, length)
            draw.line((x + dx*d/length, y + dy*d/length, x + dx*end/length, y + dy*end/length),
                      fill="#53738a", width=2)
    for item in DATA["fixtures"]:
        x, y, w, h = (item[part] for part in ("x", "y", "w", "h"))
        if item.get("checkout"):
            draw.rectangle((x, y, x + w, y + h), outline="#81769f", width=2)
            continue
        perimeter = item["id"].startswith("wall-")
        draw.rectangle((x, y, x + w, y + h), fill="#152c40" if perimeter else "#19354c",
                       outline="#395a73", width=2)
        if not perimeter:
            draw.line((x + w / 2, y, x + w / 2, y + h), fill="#37586f", width=2)
            for row in (1, 2):
                draw.line((x, y + h * row / 3, x + w, y + h * row / 3), fill="#37586f", width=2)
    for aisle in DATA["aisles"]:
        x = aisle["x"]
        draw.text((x, 82), aisle["label"].upper(), fill="#c2d4e5", anchor="mm", font_size=17)
    for node in ("entrance", "checkout"):
        x, y = point(node)
        draw.text((x, y + 18), NODES[node]["label"].upper(), anchor="mm", fill="#c2d4e5", font_size=14)
    return image


def render_map(route):
    if route is None:
        return None
    image = render_floor_plan()
    draw = ImageDraw.Draw(image)

    def point(node):
        return NODES[node]["x"], NODES[node]["y"]

    draw.line([point(node) for node in route["trace"]], fill="#70f2dd", width=5, joint="curve")
    for number, stop in enumerate(route["stops"], 1):
        x, y = point(stop)
        draw.ellipse((x-12, y-12, x+12, y+12), fill="#14796d", outline="white", width=2)
        draw.text((x, y), str(number), anchor="mm", fill="white", font_size=14)
    return image
