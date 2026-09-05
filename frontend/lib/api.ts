/**
 * API client for communicating with the SonarSight FastAPI backend.
 */

import type { DetectionReport, JobStatus, ProcessConfig } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Upload a sonar image and optional metadata file.
 * Returns the job_id for subsequent processing.
 */
export async function uploadSonarImage(
  sonarImage: File,
  metadataFile?: File | null
): Promise<{ job_id: string; message: string }> {
  const formData = new FormData();
  formData.append("sonar_image", sonarImage);
  if (metadataFile) {
    formData.append("metadata_file", metadataFile);
  }

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }

  return res.json();
}

/**
 * Trigger model inference pipeline on an uploaded job with optional dynamic threshold parameters.
 */
export async function processJob(
  jobId: string,
  config?: Partial<ProcessConfig>
): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/process/${jobId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config || {}),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Process failed");
  }

  return res.json();
}

/**
 * Poll the processing status of a job.
 */
export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/status/${jobId}`);

  if (!res.ok) {
    throw new Error("Failed to get status");
  }

  return res.json();
}

/**
 * Fetch the detection report for a completed job.
 */
export async function getReport(
  jobId: string,
  format: "json" | "csv" | "geojson" = "json"
): Promise<DetectionReport> {
  const res = await fetch(`${API_BASE}/report/${jobId}?format=${format}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get report");
  }

  return res.json();
}

/**
 * Download report as CSV blob.
 */
export async function downloadReportCSV(jobId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/report/${jobId}?format=csv`);
  if (!res.ok) throw new Error("Failed to download CSV");
  return res.blob();
}

/**
 * Download report as GeoJSON blob.
 */
export async function downloadReportGeoJSON(jobId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/report/${jobId}?format=geojson`);
  if (!res.ok) throw new Error("Failed to download GeoJSON");
  return res.blob();
}

/**
 * Get the URL for a served uploaded file.
 */
export function getUploadUrl(jobId: string, filename: string): string {
  return `${API_BASE}/uploads/${jobId}/${filename}`;
}
