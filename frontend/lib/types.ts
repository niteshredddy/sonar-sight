/**
 * Shared TypeScript types for the SonarSight Marine Tactical Command Console.
 * Structurally matches schemas/detection.py with added tactical UI extensions.
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

/** Palette modes for acoustic sonar visualization. */
export type AcousticPalette = "amber" | "copper" | "grayscale" | "emerald";

/** Threat severity levels. */
export type ThreatLevel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

/**
 * A single detected object in a sonar image.
 */
export interface Detection {
  id: string;
  class_label: ClassLabel;
  confidence: number;
  bbox: BBox;
  mask: [number, number][] | null;
  lat: number;
  lon: number;
  source_image: string;

  // Tactical calculated metrics
  estimated_length_m?: number;
  estimated_width_m?: number;
  shadow_length_m?: number;
  threat_level?: ThreatLevel;
}

/**
 * Process configuration params for real-time inference tuning.
 */
export interface ProcessConfig {
  conf_threshold: number;      // 0.1 - 0.9 (default 0.45)
  iou_threshold: number;       // 0.1 - 0.9 (default 0.45)
  min_area_pixels: number;     // 10 - 1000 px² (default 100)
  enable_nadir_mask: boolean;  // default true
  palette: AcousticPalette;    // default "amber"
}

/**
 * Full detection report for a processed job.
 */
export interface DetectionReport {
  job_id: string;
  source_file: string;
  timestamp: string;
  detections: Detection[];
}

/** Job processing status. */
export interface JobStatus {
  job_id: string;
  status: "queued" | "processing" | "done" | "error";
  message: string | null;
  progress: number | null;
}

/** Mission Telemetry Summary */
export interface TelemetryData {
  vessel_sog_kts: number;      // Speed Over Ground (knots)
  towfish_altitude_m: number;  // Altitude above seabed (meters)
  swath_width_m: number;       // Total swath width (meters)
  water_depth_m: number;       // Water depth (meters)
  active_targets: number;
  threat_score: string;        // CRITICAL, HIGH, MEDIUM, LOW
}

/** Class label tactical display configuration */
export const CLASS_CONFIG: Record<
  ClassLabel,
  { color: string; bg: string; border: string; label: string; symbol: string }
> = {
  ghost_net: {
    color: "#f59e0b",
    bg: "rgba(245, 158, 11, 0.15)",
    border: "rgba(245, 158, 11, 0.5)",
    label: "Ghost Net",
    symbol: "NET",
  },
  shipwreck: {
    color: "#ef4444",
    bg: "rgba(239, 68, 68, 0.15)",
    border: "rgba(239, 68, 68, 0.5)",
    label: "Shipwreck",
    symbol: "WRK",
  },
  pipe: {
    color: "#00d4ff",
    bg: "rgba(0, 212, 255, 0.15)",
    border: "rgba(0, 212, 255, 0.5)",
    label: "Pipeline",
    symbol: "PIP",
  },
  cylinder: {
    color: "#10b981",
    bg: "rgba(16, 185, 129, 0.15)",
    border: "rgba(16, 185, 129, 0.5)",
    label: "Cylinder/Canister",
    symbol: "CYL",
  },
};
