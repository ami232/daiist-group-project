"""Browser payload for the interactive shopping route."""

from base64 import b64encode
from functools import lru_cache
from io import BytesIO

import store


@lru_cache(maxsize=1)
def floor_plan_data_url():
    output = BytesIO()
    store.render_floor_plan().save(output, format="PNG")
    return "data:image/png;base64," + b64encode(output.getvalue()).decode("ascii")


def payload(route, matches, *, example=False):
    if not route:
        return None
    stops = [
        {
            "id": zone,
            "title": store.NODES[zone]["label"],
            "items": [{"id": match["index"], "name": match["product"]} for match in matches if match["zone"] == zone],
        }
        for zone in route["stops"]
    ]
    ordered = ["entrance", *route["stops"], "checkout"]
    positions = {node_id: (node["x"], node["y"]) for node_id, node in store.NODES.items()}
    legs = {}
    for start in ordered[:-1]:
        _, previous = store.shortest_paths(start)
        legs[start] = {
            end: [positions[node_id] for node_id in store.walking_path(previous, start, end)]
            for end in ordered[1:]
        }
    return {
        "stops": stops,
        "example": example,
        "points": {zone: positions[zone] for zone in ordered},
        "legs": legs,
        "trace": [positions[node_id] for node_id in route["trace"]],
    }


def example_payload():
    items = ["tomatoes", "pasta", "cereal", "eggs"]
    matches = [
        {"index": index, "product": store.PRODUCTS[item]["name"], "zone": store.PRODUCTS[item]["zone"]}
        for index, item in enumerate(items)
    ]
    return payload(store.plan([item["zone"] for item in matches]), matches, example=True)
