"""
Job lifecycle manager for the sonar detection backend.

Manages job creation, status tracking, and report assembly.
Uses in-memory storage (dict) suitable for hackathon demos.

For production: swap to Redis/SQLite/PostgreSQL by replacing
the JobStore class.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.detection import Detection, DetectionReport, JobStatus


class Status(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class Job:
    """Internal job representation."""

    def __init__(
        self,
        job_id: str,
        source_file: str,
        image_path: str,
        metadata_path: Optional[str] = None,
    ):
        self.job_id = job_id
        self.source_file = source_file
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.status = Status.QUEUED
        self.message: Optional[str] = None
        self.progress: float = 0.0
        self.detections: list[Detection] = []
        self.created_at = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None
        self.timings: dict = {}


class JobManager:
    """
    In-memory job manager.

    Thread-safety note: For the hackathon demo this uses a simple dict.
    For concurrent access, wrap mutations in a threading.Lock or use
    asyncio-safe storage.
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create_job(
        self,
        source_file: str,
        image_path: str,
        metadata_path: Optional[str] = None,
    ) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            source_file=source_file,
            image_path=image_path,
            metadata_path=metadata_path,
        )
        self._jobs[job_id] = job
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID."""
        return self._jobs.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: Status,
        message: Optional[str] = None,
        progress: Optional[float] = None,
    ):
        """Update job status."""
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        job.status = status
        if message is not None:
            job.message = message
        if progress is not None:
            job.progress = progress
        if status in (Status.DONE, Status.ERROR):
            job.completed_at = datetime.now(timezone.utc)

    def set_results(
        self,
        job_id: str,
        detections: list[Detection],
        timings: Optional[dict] = None,
    ):
        """Store detection results for a completed job."""
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        job.detections = detections
        if timings:
            job.timings = timings

    def get_status(self, job_id: str) -> JobStatus:
        """Get the current status of a job."""
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        return JobStatus(
            job_id=job.job_id,
            status=job.status.value,
            message=job.message,
            progress=job.progress,
        )

    def get_report(self, job_id: str) -> DetectionReport:
        """Assemble and return the detection report for a completed job."""
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        return DetectionReport(
            job_id=job.job_id,
            source_file=job.source_file,
            timestamp=job.completed_at or job.created_at,
            detections=job.detections,
        )

    def list_jobs(self) -> list[dict]:
        """List all jobs with basic info."""
        return [
            {
                "job_id": j.job_id,
                "source_file": j.source_file,
                "status": j.status.value,
                "created_at": j.created_at.isoformat(),
                "num_detections": len(j.detections),
            }
            for j in self._jobs.values()
        ]
