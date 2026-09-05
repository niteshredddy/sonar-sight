/**
 * Shared TypeScript interfaces for the Marine Debris Detection System.
 * These MUST remain structurally identical to schemas/detection.py.
 * Any changes here MUST be mirrored in the Python Pydantic models.
 */

/** Axis-aligned bounding box in pixel coordinates. */
export interface BBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

/** Valid class labels for detection targets. */
export type ClassLabel = "ghost_net" | "shipwreck" | "pipe" | "cylinder";

/**
 * A single detected object in a sonar image.
 * Coordinates are in the original (untiled) image space.
 * Confidence is on a 0-100 scale (post-rescoring).
 */
export interface Detection {
  id: string;
  class_label: ClassLabel;
  confidence: number; // 0-100
  bbox: BBox;
  mask: [number, number][] | null; // Optional polygon vertices
  lat: number;
  lon: number;
  source_image: string;
}

/**
 * Assembled report for a single processing job.
 * Contains all detections found in the uploaded sonar image.
 */
export interface DetectionReport {
  job_id: string;
  source_file: string;
  timestamp: string; // ISO 8601
  detections: Detection[];
}

/** Job processing status. */
export interface JobStatus {
  job_id: string;
  status: "queued" | "processing" | "done" | "error";
  message: string | null;
  progress: number | null; // 0-100
}

/** Per-ping navigation data for geotagging. */
export interface PingMetadata {
  ping_index: number;
  lat: number;
  lon: number;
  heading: number; // degrees from north, 0-360
  altitude: number; // meters above seabed
  timestamp?: string;
}

/** Metadata accompanying a sonar image upload. */
export interface UploadMetadata {
  pings: PingMetadata[] | null;
}
