"""Shared geometry / geo helpers."""
from __future__ import annotations

import math

from config import GOAL_X, GOAL_Y, GOAL_WIDTH, HOST_CITIES


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def city_distance_km(city_a: str, city_b: str) -> float:
    """Travel distance between two host cities (0 if same or unknown)."""
    if city_a == city_b or city_a not in HOST_CITIES or city_b not in HOST_CITIES:
        return 0.0
    a, b = HOST_CITIES[city_a], HOST_CITIES[city_b]
    return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])


def shot_geometry(x: float, y: float) -> tuple[float, float]:
    """Return (distance_to_goal_centre, goal-mouth angle in radians).

    Angle is the angle subtended by the two goalposts from the shot location —
    the standard geometric driver of xG. Larger angle => easier chance.
    """
    dx = GOAL_X - x
    distance = math.hypot(dx, GOAL_Y - y)

    post1 = (GOAL_X, GOAL_Y - GOAL_WIDTH / 2)
    post2 = (GOAL_X, GOAL_Y + GOAL_WIDTH / 2)
    a = math.hypot(post1[0] - x, post1[1] - y)
    b = math.hypot(post2[0] - x, post2[1] - y)
    c = GOAL_WIDTH
    # Law of cosines for the angle at the shot vertex.
    cos_theta = (a * a + b * b - c * c) / (2 * a * b + 1e-9)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle = math.acos(cos_theta)
    return distance, angle
