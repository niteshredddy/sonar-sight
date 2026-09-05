"""
Unit tests for the geotagging module.

Tests slant-range correction, track interpolation, and pixel-to-lat/lon
conversion using a synthetic straight-line northward track with known
expected positions.
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.detection import BBox, Detection, PingMetadata
from backend.geotag import (
    geotag_detection,
    geotag_detections,
    interpolate_track_position,
    offset_lat_lon,
    pixel_to_across_track_meters,
    slant_to_ground_range,
)


# ─── Slant-range correction tests ────────────────────────────────────────────


class TestSlantToGroundRange:
    def test_zero_slant(self):
        """Zero slant range → zero ground range."""
        assert slant_to_ground_range(0, 10) == 0.0

    def test_slant_equals_altitude(self):
        """Slant range equal to altitude → directly below → zero ground range."""
        assert slant_to_ground_range(10, 10) == 0.0

    def test_slant_less_than_altitude(self):
        """Slant range less than altitude → clamp to zero."""
        assert slant_to_ground_range(5, 10) == 0.0

    def test_known_triangle(self):
        """Classic 3-4-5 triangle: slant=5, altitude=3 → ground=4."""
        result = slant_to_ground_range(5, 3)
        assert abs(result - 4.0) < 1e-10

    def test_known_triangle_2(self):
        """Slant=13, altitude=5 → ground=12."""
        result = slant_to_ground_range(13, 5)
        assert abs(result - 12.0) < 1e-10

    def test_large_slant(self):
        """Large slant range with typical altitude."""
        result = slant_to_ground_range(75, 10)
        expected = math.sqrt(75**2 - 10**2)
        assert abs(result - expected) < 1e-10


# ─── Pixel to across-track tests ────────────────────────────────────────────


class TestPixelToAcrossTrack:
    def test_center_pixel(self):
        """Center pixel → nadir → zero across-track distance."""
        result = pixel_to_across_track_meters(320, 640, max_slant_range=75, altitude=10)
        # At nadir, slant_range ≈ 0, so ground_range ≈ 0
        assert abs(result) < 0.5  # Nearly zero

    def test_port_starboard_symmetry(self):
        """Equidistant port and starboard pixels should have equal magnitude, opposite sign."""
        port = pixel_to_across_track_meters(100, 640, max_slant_range=75, altitude=10)
        starboard = pixel_to_across_track_meters(540, 640, max_slant_range=75, altitude=10)
        assert port < 0  # Port side (left)
        assert starboard > 0  # Starboard side (right)
        assert abs(abs(port) - abs(starboard)) < 0.5

    def test_edge_pixels(self):
        """Edge pixels should give maximum across-track distance."""
        left = pixel_to_across_track_meters(0, 640, max_slant_range=75, altitude=10)
        right = pixel_to_across_track_meters(639, 640, max_slant_range=75, altitude=10)
        max_ground = slant_to_ground_range(75, 10)
        assert abs(abs(left) - max_ground) < 1.0
        assert abs(abs(right) - max_ground) < 1.0


# ─── Track interpolation tests ──────────────────────────────────────────────


class TestTrackInterpolation:
    @pytest.fixture
    def straight_track(self):
        """Create a simple northward straight-line track."""
        pings = []
        for i in range(100):
            pings.append(PingMetadata(
                ping_index=i,
                lat=19.0 + i * 0.0001,  # Moving north
                lon=72.0,
                heading=0.0,
                altitude=10.0,
            ))
        return pings

    def test_exact_ping(self, straight_track):
        """Exact ping index should return that ping's coordinates."""
        lat, lon, heading = interpolate_track_position(50, straight_track)
        assert abs(lat - (19.0 + 50 * 0.0001)) < 1e-10
        assert abs(lon - 72.0) < 1e-10
        assert abs(heading - 0.0) < 1e-10

    def test_interpolated_ping(self, straight_track):
        """Fractional ping index should interpolate between neighbors."""
        lat, lon, heading = interpolate_track_position(50.5, straight_track)
        expected_lat = 19.0 + 50.5 * 0.0001
        assert abs(lat - expected_lat) < 1e-8

    def test_first_ping(self, straight_track):
        """Index before first ping should clamp to first."""
        lat, _, _ = interpolate_track_position(-5, straight_track)
        assert abs(lat - 19.0) < 1e-10

    def test_last_ping(self, straight_track):
        """Index after last ping should clamp to last."""
        lat, _, _ = interpolate_track_position(200, straight_track)
        assert abs(lat - (19.0 + 99 * 0.0001)) < 1e-10

    def test_heading_interpolation(self):
        """Heading interpolation should handle wrap-around at 360°."""
        pings = [
            PingMetadata(ping_index=0, lat=19.0, lon=72.0, heading=350.0, altitude=10),
            PingMetadata(ping_index=10, lat=19.0, lon=72.0, heading=10.0, altitude=10),
        ]
        _, _, heading = interpolate_track_position(5, pings)
        assert abs(heading - 0.0) < 1.0 or abs(heading - 360.0) < 1.0


# ─── Offset lat/lon tests ───────────────────────────────────────────────────


class TestOffsetLatLon:
    def test_zero_offset(self):
        """Zero offset should return the same position."""
        lat, lon = offset_lat_lon(19.0, 72.0, 0.0, 0.0)
        assert abs(lat - 19.0) < 1e-10
        assert abs(lon - 72.0) < 1e-10

    def test_starboard_offset_heading_north(self):
        """Heading north (0°), starboard offset (+) should move east (lon increases)."""
        lat, lon = offset_lat_lon(19.0, 72.0, 0.0, 50.0)
        assert abs(lat - 19.0) < 0.001  # Minimal lat change
        assert lon > 72.0  # Should move east

    def test_port_offset_heading_north(self):
        """Heading north (0°), port offset (-) should move west (lon decreases)."""
        lat, lon = offset_lat_lon(19.0, 72.0, 0.0, -50.0)
        assert abs(lat - 19.0) < 0.001
        assert lon < 72.0  # Should move west

    def test_offset_magnitude(self):
        """50 meters offset should give roughly 50m displacement."""
        lat0, lon0 = 19.0, 72.0
        lat1, lon1 = offset_lat_lon(lat0, lon0, 0.0, 50.0)

        # Convert back to meters approximately
        m_per_deg_lon = 111_320 * math.cos(math.radians(lat0))
        dist_m = abs(lon1 - lon0) * m_per_deg_lon
        assert abs(dist_m - 50.0) < 1.0  # Within 1 meter


# ─── Full geotagging test ───────────────────────────────────────────────────


class TestGeotagDetection:
    def test_center_detection_on_straight_track(self):
        """A detection at the image center on a straight northward track
        should be geotagged near the mid-track position, near nadir."""
        pings = []
        for i in range(640):
            pings.append(PingMetadata(
                ping_index=i,
                lat=19.0 + i * 0.000001,
                lon=72.0,
                heading=0.0,
                altitude=10.0,
            ))

        detection = Detection(
            class_label="cylinder",
            confidence=85.0,
            bbox=BBox(x_min=300, y_min=300, x_max=340, y_max=340),
            lat=0.0, lon=0.0,
            source_image="test.png",
        )

        result = geotag_detection(
            detection, pings,
            image_height=640, image_width=640,
            max_slant_range=75.0,
        )

        # Should be roughly in the middle of the track
        assert 19.0 < result.lat < 19.001
        # Should be very close to the track line (center pixel ≈ nadir)
        assert abs(result.lon - 72.0) < 0.01

    def test_valid_lat_lon_range(self):
        """All geotagged detections should have valid lat/lon."""
        pings = [
            PingMetadata(ping_index=0, lat=19.0, lon=72.0, heading=0.0, altitude=10.0),
            PingMetadata(ping_index=639, lat=19.01, lon=72.0, heading=0.0, altitude=10.0),
        ]

        detections = [
            Detection(
                class_label="pipe",
                confidence=70.0,
                bbox=BBox(x_min=50, y_min=100, x_max=200, y_max=120),
                lat=0.0, lon=0.0,
                source_image="test.png",
            ),
            Detection(
                class_label="shipwreck",
                confidence=90.0,
                bbox=BBox(x_min=400, y_min=500, x_max=550, y_max=560),
                lat=0.0, lon=0.0,
                source_image="test.png",
            ),
        ]

        results = geotag_detections(detections, pings, 640, 640)

        for det in results:
            assert -90 <= det.lat <= 90
            assert -180 <= det.lon <= 180
            assert det.lat != 0.0 or det.lon != 0.0  # Should be geotagged
