"""
Geotagging module for side-scan sonar detections.

Converts each detection's bounding box center from pixel space to
real-world latitude/longitude using slant-range geometry:

    Along-track:  pixel row → ping index → interpolated (lat, lon) on track
    Across-track: pixel col → slant range → ground range → perpendicular offset

Slant-range to ground-range correction:
    G_r = sqrt(S_r² - H²)    where H = towfish altitude above seabed

Coordinate system:
    - Image rows = along-track (ping index, sequential in time)
    - Image columns = across-track (range, distance from nadir)
    - Left half = port side, right half = starboard side
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.detection import BBox, Detection, PingMetadata

# Earth radius in meters (WGS84 mean)
EARTH_RADIUS_M = 6_371_000.0


def slant_to_ground_range(slant_range: float, altitude: float) -> float:
    """
    Convert slant range to ground range using flat-floor assumption.

    G_r = sqrt(S_r² - H²)

    Args:
        slant_range: Distance from transducer to target (meters)
        altitude: Towfish altitude above seabed (meters)

    Returns:
        Horizontal distance from nadir to target (meters)
    """
    if slant_range <= altitude:
        return 0.0
    return math.sqrt(slant_range ** 2 - altitude ** 2)


def pixel_to_across_track_meters(
    pixel_col: float,
    image_width: int,
    max_slant_range: float = 75.0,
    altitude: float = 10.0,
) -> float:
    """
    Convert pixel column to across-track distance in meters.

    Assumes the sonar image covers port-to-starboard from left to right,
    with nadir at the center column.

    Args:
        pixel_col: Column position in the image
        image_width: Total image width in pixels
        max_slant_range: Maximum slant range of the sonar (meters)
        altitude: Towfish altitude (meters)

    Returns:
        Across-track distance from nadir (meters). Negative = port, positive = starboard.
    """
    # Normalize pixel position: 0 = left edge, 1 = right edge
    normalized = pixel_col / max(image_width - 1, 1)

    # Map to slant range: -max_slant_range (port) to +max_slant_range (starboard)
    # Center (0.5) = nadir
    slant_range = (normalized - 0.5) * 2.0 * max_slant_range

    # Determine sign (port/starboard) and convert to ground range
    sign = 1.0 if slant_range >= 0 else -1.0
    ground_range = slant_to_ground_range(abs(slant_range), altitude)

    return sign * ground_range


def interpolate_track_position(
    ping_index: float,
    pings: list[PingMetadata],
) -> tuple[float, float, float]:
    """
    Interpolate lat/lon/heading at a fractional ping index.

    Uses linear interpolation between adjacent ping records.

    Returns:
        (lat, lon, heading) at the interpolated position
    """
    if not pings:
        return (0.0, 0.0, 0.0)

    if len(pings) == 1:
        return (pings[0].lat, pings[0].lon, pings[0].heading)

    # Sort pings by index
    sorted_pings = sorted(pings, key=lambda p: p.ping_index)

    # Clamp to valid range
    if ping_index <= sorted_pings[0].ping_index:
        return (sorted_pings[0].lat, sorted_pings[0].lon, sorted_pings[0].heading)
    if ping_index >= sorted_pings[-1].ping_index:
        return (sorted_pings[-1].lat, sorted_pings[-1].lon, sorted_pings[-1].heading)

    # Find bracketing pings
    for i in range(len(sorted_pings) - 1):
        p0 = sorted_pings[i]
        p1 = sorted_pings[i + 1]

        if p0.ping_index <= ping_index <= p1.ping_index:
            # Linear interpolation factor
            span = p1.ping_index - p0.ping_index
            if span == 0:
                t = 0.0
            else:
                t = (ping_index - p0.ping_index) / span

            lat = p0.lat + t * (p1.lat - p0.lat)
            lon = p0.lon + t * (p1.lon - p0.lon)

            # Interpolate heading (handle wrap-around at 360)
            h0 = p0.heading
            h1 = p1.heading
            diff = h1 - h0
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            heading = (h0 + t * diff) % 360

            return (lat, lon, heading)

    # Fallback
    return (sorted_pings[-1].lat, sorted_pings[-1].lon, sorted_pings[-1].heading)


def offset_lat_lon(
    lat: float,
    lon: float,
    heading: float,
    across_track_m: float,
) -> tuple[float, float]:
    """
    Offset a lat/lon position perpendicular to the heading.

    Across-track offset is applied perpendicular to the vessel's heading:
        - Positive across_track_m = starboard (heading + 90°)
        - Negative across_track_m = port (heading - 90°)

    Uses simple flat-earth approximation (valid for short distances < ~1km).

    Args:
        lat: Base latitude (degrees)
        lon: Base longitude (degrees)
        heading: Vessel heading in degrees from north (0=N, 90=E)
        across_track_m: Perpendicular distance in meters

    Returns:
        (new_lat, new_lon) offset position
    """
    # Perpendicular direction (starboard = heading + 90°)
    perp_heading_deg = (heading + 90.0) % 360.0
    perp_heading_rad = math.radians(perp_heading_deg)

    # Convert distance to angular offset
    lat_rad = math.radians(lat)

    # Meters per degree at this latitude
    m_per_deg_lat = EARTH_RADIUS_M * math.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * math.cos(lat_rad)

    # Offset in north/east components
    d_north = across_track_m * math.cos(perp_heading_rad)
    d_east = across_track_m * math.sin(perp_heading_rad)

    # Convert to degrees
    new_lat = lat + d_north / m_per_deg_lat
    new_lon = lon + d_east / max(m_per_deg_lon, 1e-10)

    return (new_lat, new_lon)


def geotag_detection(
    detection: Detection,
    pings: list[PingMetadata],
    image_height: int,
    image_width: int,
    max_slant_range: float = 75.0,
) -> Detection:
    """
    Assign real-world lat/lon to a detection based on its pixel position.

    Args:
        detection: Detection object (bbox center used for geotagging)
        pings: List of per-ping navigation metadata
        image_height: Source image height in pixels (= number of pings)
        image_width: Source image width in pixels (= range samples)
        max_slant_range: Maximum sonar range in meters

    Returns:
        Updated Detection with lat/lon filled in
    """
    if not pings:
        return detection

    # Get bbox center
    cx, cy = detection.bbox.center

    # Along-track: pixel row → ping index
    # Map image row to ping index range
    min_ping = min(p.ping_index for p in pings)
    max_ping = max(p.ping_index for p in pings)
    ping_range = max_ping - min_ping

    if ping_range > 0 and image_height > 1:
        ping_index = min_ping + (cy / (image_height - 1)) * ping_range
    else:
        ping_index = min_ping

    # Get interpolated track position
    lat, lon, heading = interpolate_track_position(ping_index, pings)

    # Get altitude from nearest ping
    nearest_ping = min(pings, key=lambda p: abs(p.ping_index - ping_index))
    altitude = nearest_ping.altitude

    # Across-track: pixel col → ground range → perpendicular offset
    across_track_m = pixel_to_across_track_meters(
        cx, image_width, max_slant_range, altitude
    )

    # Apply offset to get detection lat/lon
    det_lat, det_lon = offset_lat_lon(lat, lon, heading, across_track_m)

    # Update detection
    detection.lat = round(det_lat, 7)
    detection.lon = round(det_lon, 7)

    return detection


def geotag_detections(
    detections: list[Detection],
    pings: list[PingMetadata],
    image_height: int,
    image_width: int,
    max_slant_range: float = 75.0,
) -> list[Detection]:
    """Geotag a list of detections."""
    return [
        geotag_detection(d, pings, image_height, image_width, max_slant_range)
        for d in detections
    ]
