"""
FastAPI backend for the SonarSight marine debris detection system.

Endpoints:
    POST /upload         — Upload sonar image + metadata, returns job_id
    POST /process/{id}   — Trigger model inference pipeline
    GET  /status/{id}    — Get job processing status
    GET  /report/{id}    — Get detection report (JSON or CSV)
    GET  /uploads/{id}/{f} — Serve uploaded files for frontend

Run:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import csv
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.detection import DetectionReport, JobStatus, PingMetadata
from backend.geotag import geotag_detections
from backend.job_manager import JobManager, Status

# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SonarSight API",
    description="Marine debris detection in side-scan sonar imagery",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Job manager
job_manager = JobManager()

# Lazy-load inference pipeline (heavy imports)
_pipeline = None


def get_pipeline():
    """Lazy-load the inference pipeline to avoid slow startup."""
    global _pipeline
    if _pipeline is None:
        from model.inference import SonarInferencePipeline

        # Try to find trained model, fall back to mock mode
        model_path = None
        for candidate in [
            PROJECT_ROOT / "runs" / "sonar" / "train" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "sonar" / "train" / "weights" / "best.onnx",
        ]:
            if candidate.exists():
                model_path = str(candidate)
                break

        _pipeline = SonarInferencePipeline(
            model_path=model_path,
            tile_size=640,
            tile_overlap=0.5,
        )
    return _pipeline


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Health check."""
    return {
        "service": "SonarSight API",
        "status": "running",
        "version": "1.0.0",
    }


@app.post("/upload")
async def upload_sonar(
    sonar_image: UploadFile = File(...),
    metadata_file: Optional[UploadFile] = File(None),
):
    """
    Upload a sonar image and optional navigation metadata file.

    The metadata file should be a JSON or CSV with per-ping lat/lon/heading data.
    Returns a job_id for subsequent processing.
    """
    # Validate file type
    allowed_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    ext = Path(sonar_image.filename or "image.png").suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported image format: {ext}")

    # Create job
    job_id = job_manager.create_job(
        source_file=sonar_image.filename or "unknown",
        image_path="",  # Will update after saving
        metadata_path=None,
    )

    # Save files
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / (sonar_image.filename or "sonar_image.png")
    content = await sonar_image.read()
    image_path.write_bytes(content)

    metadata_path = None
    if metadata_file and metadata_file.filename:
        metadata_path = job_dir / metadata_file.filename
        meta_content = await metadata_file.read()
        metadata_path.write_bytes(meta_content)

    # Update job with file paths
    job = job_manager.get_job(job_id)
    if job:
        job.image_path = str(image_path)
        job.metadata_path = str(metadata_path) if metadata_path else None

    return {"job_id": job_id, "message": "Upload successful. Call POST /process/{job_id} to start."}


class ProcessParams(BaseModel):
    conf_threshold: float = 0.45
    iou_threshold: float = 0.45
    min_area_pixels: int = 100
    enable_nadir_mask: bool = True


@app.post("/process/{job_id}")
async def process_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    params: Optional[ProcessParams] = None,
):
    """
    Trigger model inference pipeline on an uploaded sonar image with optional thresholds.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    if job.status != Status.QUEUED and job.status != Status.DONE:
        raise HTTPException(400, f"Job status is {job.status.value}")

    job_manager.update_status(job_id, Status.PROCESSING, "Starting inference pipeline...")
    process_config = params.dict() if params else {}
    background_tasks.add_task(_run_pipeline, job_id, process_config)

    return {"job_id": job_id, "status": "processing"}


def _run_pipeline(job_id: str, process_config: dict):
    """Background task: run the full inference + geotagging pipeline."""
    try:
        job = job_manager.get_job(job_id)
        if not job:
            return

        pipeline = get_pipeline()

        # Load image
        job_manager.update_status(job_id, Status.PROCESSING, "Loading image...", progress=10)
        image = cv2.imread(job.image_path)
        if image is None:
            raise ValueError(f"Could not read image: {job.image_path}")

        image_h, image_w = image.shape[:2]

        # Run inference with dynamic thresholds
        job_manager.update_status(job_id, Status.PROCESSING, "Running detection model...", progress=30)
        detections, timings = pipeline.run_on_array(
            image,
            source_name=job.source_file,
            conf_threshold=process_config.get("conf_threshold", 0.45),
            iou_threshold=process_config.get("iou_threshold", 0.45),
            min_area_pixels=process_config.get("min_area_pixels", 100),
            enable_nadir_mask=process_config.get("enable_nadir_mask", True),
        )

        # Load and parse metadata for geotagging
        job_manager.update_status(job_id, Status.PROCESSING, "Geotagging detections...", progress=70)
        pings = _load_metadata(job.metadata_path, image_h)

        if pings:
            detections = geotag_detections(
                detections, pings, image_h, image_w, max_slant_range=75.0
            )

        # Store results
        job_manager.update_status(job_id, Status.PROCESSING, "Assembling report...", progress=90)
        job_manager.set_results(job_id, detections, timings)
        job_manager.update_status(
            job_id, Status.DONE,
            f"Complete: {len(detections)} detections found",
            progress=100,
        )

    except Exception as e:
        traceback.print_exc()
        job_manager.update_status(job_id, Status.ERROR, f"Pipeline error: {str(e)}")


def _load_metadata(
    metadata_path: Optional[str],
    image_height: int,
) -> list[PingMetadata]:
    """Load navigation metadata from JSON or CSV file."""
    if not metadata_path or not Path(metadata_path).exists():
        return _generate_demo_track(image_height)

    path = Path(metadata_path)

    try:
        if path.suffix.lower() == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return [PingMetadata(**p) for p in data]
            elif isinstance(data, dict) and "pings" in data:
                return [PingMetadata(**p) for p in data["pings"]]

        elif path.suffix.lower() == ".csv":
            pings = []
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pings.append(PingMetadata(
                        ping_index=int(row.get("ping_index", row.get("index", 0))),
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                        heading=float(row.get("heading", 0)),
                        altitude=float(row.get("altitude", 10)),
                    ))
            return pings

    except Exception as e:
        print(f"⚠️ Could not parse metadata: {e}. Using demo track.")

    return _generate_demo_track(image_height)


def _generate_demo_track(
    num_pings: int,
    start_lat: float = 18.9250,   # Offshore Arabian Sea (off Mumbai coast)
    start_lon: float = 72.6800,
    heading: float = 0.0,         # Due north
    ping_spacing_m: float = 0.5,  # 50cm between pings
) -> list[PingMetadata]:
    """Generate a synthetic straight-line track in offshore Arabian Sea waters."""
    import math

    pings = []
    lat = start_lat
    lon = start_lon

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))

    heading_rad = math.radians(heading)
    d_lat = (ping_spacing_m * math.cos(heading_rad)) / m_per_deg_lat
    d_lon = (ping_spacing_m * math.sin(heading_rad)) / m_per_deg_lon

    for i in range(num_pings):
        pings.append(PingMetadata(
            ping_index=i,
            lat=round(lat + i * d_lat, 7),
            lon=round(lon + i * d_lon, 7),
            heading=heading,
            altitude=12.5,
        ))

    return pings


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the current processing status of a job."""
    try:
        status = job_manager.get_status(job_id)
        return status.model_dump()
    except KeyError:
        raise HTTPException(404, f"Job not found: {job_id}")


@app.get("/report/{job_id}")
async def get_report(
    job_id: str,
    format: str = Query("json", pattern="^(json|csv|geojson)$"),
):
    """
    Get the detection report for a completed job.

    Query params:
        format: "json" (default), "csv", or "geojson"
    """
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")

        if job.status != Status.DONE:
            raise HTTPException(400, f"Job not complete. Status: {job.status.value}")

        report = job_manager.get_report(job_id)

        if format == "csv":
            return _report_to_csv_response(report)
        elif format == "geojson":
            return _report_to_geojson(report)
        else:
            return report.model_dump(mode="json")

    except KeyError:
        raise HTTPException(404, f"Job not found: {job_id}")


def _report_to_geojson(report: DetectionReport) -> dict:
    """Convert a DetectionReport to a GeoJSON FeatureCollection."""
    features = []
    for det in report.detections:
        # Determine threat level
        if det.confidence >= 80:
            threat = "HIGH"
        elif det.confidence >= 50:
            threat = "MEDIUM"
        else:
            threat = "LOW"

        # Calculate estimated physical dimensions based on acoustic geometry
        bbox_w = abs(det.bbox.x_max - det.bbox.x_min)
        bbox_h = abs(det.bbox.y_max - det.bbox.y_min)

        length_m = round(max(bbox_w, bbox_h) * 0.12, 1)  # ~12cm/px resolution
        width_m = round(min(bbox_w, bbox_h) * 0.12, 1)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [det.lon, det.lat],
            },
            "properties": {
                "id": det.id,
                "class_label": det.class_label,
                "confidence": det.confidence,
                "bbox": [det.bbox.x_min, det.bbox.y_min, det.bbox.x_max, det.bbox.y_max],
                "threat_level": threat,
                "estimated_length_m": length_m,
                "estimated_width_m": width_m,
                "source_image": det.source_image,
            },
        })

    return {
        "type": "FeatureCollection",
        "job_id": report.job_id,
        "source_file": report.source_file,
        "timestamp": report.timestamp,
        "features": features,
    }


def _report_to_csv_response(report: DetectionReport) -> StreamingResponse:
    """Convert a DetectionReport to a downloadable CSV response."""
    rows = report.to_csv_rows()

    if not rows:
        # Empty report
        output = io.StringIO()
        output.write("No detections found\n")
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=report_{report.job_id}.csv"},
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=report_{report.job_id}.csv"},
    )


@app.get("/uploads/{job_id}/{filename}")
async def serve_upload(job_id: str, filename: str):
    """Serve uploaded files (images) for frontend display."""
    file_path = UPLOAD_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(file_path)


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return job_manager.list_jobs()
