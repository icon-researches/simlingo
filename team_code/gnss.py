"""CARLA sensor GNSS and the bundled leaderboard's navigation GPS projections.

CARLA 0.9.16 changed sensor GNSS to spherical transverse Mercator (including
the direction of world y). The bundled leaderboard still generates route GPS
with the older Mercator formula. Keep those two conversions separate.

These helpers need neither a simulator connection nor access to the ego actor.
"""

import math

import numpy as np
from scipy.optimize import least_squares

EARTH_RADIUS = 6378137.0


def gps_to_world(gps, lat_ref, lon_ref):
    """Invert the legacy Mercator projection used by navigation plans."""
    lat, lon, altitude = gps
    scale = math.cos(math.radians(lat_ref))
    x = scale * EARTH_RADIUS * math.radians(lon - lon_ref)
    y = (
        scale
        * EARTH_RADIUS
        * (
            math.log(math.tan(math.radians(90 + lat_ref) / 2))
            - math.log(math.tan(math.radians(90 + lat) / 2))
        )
    )
    return np.array([x, y, altitude])


def gnss_to_world(gps, lat_ref, lon_ref):
    """Invert CARLA 0.9.16 sensor GNSS.

    CARLA 0.9.16 uses GeoLocation::GetTransversemercProjection: spherical
    transverse Mercator with k0=0.9996 and world +y northward.
    Altitude is passed through, as in the route planner (benchmark maps have
    zero reference altitude).
    """
    lat, lon, altitude = gps
    phi = math.radians(lat)
    delta_lon = math.radians(lon - lon_ref)
    b = math.cos(phi) * math.sin(delta_lon)
    radius = EARTH_RADIUS * 0.9996
    x = radius * math.atanh(b)
    y = radius * (
        math.atan(math.tan(phi) / math.cos(delta_lon)) - math.radians(lat_ref)
    )
    return np.array([x, y, altitude])


def gps_reference(gps, world_xy):
    """Recover the map reference from a legacy navigation GPS/world pair.

    A failed fit stops evaluation instead of silently using a zero reference,
    which would corrupt localization on maps with a nonzero reference.
    """
    world_xy = np.asarray(world_xy, dtype=float)
    result = least_squares(
        lambda ref: gps_to_world(gps, *ref)[:2] - world_xy,
        x0=np.asarray(gps[:2], dtype=float),
        bounds=([-85, -180], [85, 180]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    if not result.success or np.linalg.norm(result.fun) > 0.01:
        raise ValueError(f"Cannot calibrate the route GNSS reference: {result.message}")
    return tuple(result.x)
