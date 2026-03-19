"""Geodetic conversion helpers."""

from __future__ import annotations

import math

from aeosbench.constants import RADIUS_EARTH


def lla_to_pcpf(
    latitude_radians: float,
    longitude_radians: float,
    altitude: float = 0.0,
    planet_radius: float = RADIUS_EARTH,
) -> tuple[float, float, float]:
    """Convert lat/lon/alt into planet-centered, planet-fixed coordinates."""

    s_phi = math.sin(latitude_radians)
    n_value = planet_radius / math.sqrt(1.0 - 0.0 * s_phi * s_phi)
    return (
        (n_value + altitude) * math.cos(latitude_radians) * math.cos(longitude_radians),
        (n_value + altitude) * math.cos(latitude_radians) * math.sin(longitude_radians),
        n_value * s_phi,
    )
