"""
Shared Pydantic schemas for the Marine Debris Detection System.
These models define the contract between the model pipeline, backend API,
and frontend. Any changes here MUST be mirrored in schemas/detection.ts.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height


# Valid class labels for detection targets
CLASS_LABELS = ["ghost_net", "shipwreck", "pipe", "cylinder"]


class Detection(BaseModel):
    """
    A single detected object in a sonar image.
    
    Coordinates are in the original (untiled) image space.
    Confidence is on a 0-100 scale (post-rescoring).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    class_label: str = Field(..., description="One of: ghost_net, shipwreck, pipe, cylinder")
    confidence: float = Field(..., ge=0, le=100, description="Adjusted confidence score 0-100")
    bbox: BBox = Field(..., description="Bounding box in original image coordinates")
    mask: Optional[list[tuple[float, float]]] = Field(
        default=None,
        description="Optional polygon mask as list of (x, y) vertices"
    )
    lat: float = Field(..., ge=-90, le=90, description="Latitude of detection center")
    lon: float = Field(..., ge=-180, le=180, description="Longitude of detection center")
    source_image: str = Field(..., description="Filename of the source sonar image")


class DetectionReport(BaseModel):
    """
    Assembled report for a single processing job.
    Contains all detections found in the uploaded sonar image.
    """
    job_id: str = Field(..., description="Unique job identifier")
    source_file: str = Field(..., description="Original uploaded filename")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp (UTC)"
    )
    detections: list[Detection] = Field(
        default_factory=list,
        description="List of all detections in this job"
    )

    def to_csv_rows(self) -> list[dict]:
        """Flatten detections into CSV-compatible dicts."""
        rows = []
        for det in self.detections:
            rows.append({
                "job_id": self.job_id,
                "source_file": self.source_file,
                "timestamp": self.timestamp.isoformat(),
                "detection_id": det.id,
                "class_label": det.class_label,
                "confidence": det.confidence,
                "bbox_x_min": det.bbox.x_min,
                "bbox_y_min": det.bbox.y_min,
                "bbox_x_max": det.bbox.x_max,
                "bbox_y_max": det.bbox.y_max,
                "lat": det.lat,
                "lon": det.lon,
                "source_image": det.source_image,
                "mask": str(det.mask) if det.mask else "",
            })
        return rows


class JobStatus(BaseModel):
    """Job processing status."""
    job_id: str
    status: str = Field(..., description="One of: queued, processing, done, error")
    message: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")


class UploadMetadata(BaseModel):
    """Metadata accompanying a sonar image upload."""
    pings: Optional[list[PingMetadata]] = None


class PingMetadata(BaseModel):
    """Per-ping navigation data for geotagging."""
    ping_index: int
    lat: float
    lon: float
    heading: float = Field(..., ge=0, lt=360, description="Heading in degrees from north")
    altitude: float = Field(default=10.0, ge=0, description="Towfish altitude above seabed in meters")
    timestamp: Optional[str] = None
