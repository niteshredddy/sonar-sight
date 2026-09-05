"""Schemas package - shared models for the Marine Debris Detection System."""

from .detection import (
    BBox,
    CLASS_LABELS,
    Detection,
    DetectionReport,
    JobStatus,
    PingMetadata,
    UploadMetadata,
)

__all__ = [
    "BBox",
    "CLASS_LABELS",
    "Detection",
    "DetectionReport",
    "JobStatus",
    "PingMetadata",
    "UploadMetadata",
]
