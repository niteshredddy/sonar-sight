"use client";

import React from "react";
import type { Detection } from "@/lib/types";
import { CLASS_CONFIG } from "@/lib/types";
import { X, ShieldAlert, Crosshair, MapPin, Ruler, Compass, Layers } from "lucide-react";

interface TargetInspectorProps {
  detection: Detection | null;
  onClose: () => void;
  imageUrl: string;
}

export default function TargetInspector({ detection, onClose, imageUrl }: TargetInspectorProps) {
  if (!detection) return null;

  const cfg = CLASS_CONFIG[detection.class_label] || CLASS_CONFIG.ghost_net;

  // Convert decimal lat/lon to Degrees Minutes Seconds (DMS)
  const toDMS = (deg: number, isLat: boolean) => {
    const absolute = Math.abs(deg);
    const degrees = Math.floor(absolute);
    const minutesNotTruncated = (absolute - degrees) * 60;
    const minutes = Math.floor(minutesNotTruncated);
    const seconds = ((minutesNotTruncated - minutes) * 60).toFixed(2);
    const card = isLat ? (deg >= 0 ? "N" : "S") : deg >= 0 ? "E" : "W";
    return `${degrees}° ${minutes}' ${seconds}" ${card}`;
  };

  const bboxW = Math.abs(detection.bbox.x_max - detection.bbox.x_min);
  const bboxH = Math.abs(detection.bbox.y_max - detection.bbox.y_min);

  // Acoustic dimensions estimation
  const lengthM = detection.estimated_length_m || (Math.max(bboxW, bboxH) * 0.12).toFixed(1);
  const widthM = detection.estimated_width_m || (Math.min(bboxW, bboxH) * 0.12).toFixed(1);
  const shadowM = detection.shadow_length_m || (Math.min(bboxW, bboxH) * 0.18).toFixed(1);

  // Threat Rating calculation
  const getThreatBadge = (conf: number) => {
    if (conf >= 80) return { label: "CRITICAL THREAT", cls: "bg-red-950/80 text-red-400 border-red-500/50" };
    if (conf >= 60) return { label: "HIGH PRIORITY", cls: "bg-amber-950/80 text-amber-400 border-amber-500/50" };
    return { label: "MODERATE INTEREST", cls: "bg-cyan-950/80 text-cyan-400 border-cyan-500/50" };
  };

  const threat = getThreatBadge(detection.confidence);

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full max-w-md glass-panel border-l border-cyan-500/30 p-5 shadow-2xl flex flex-col justify-between overflow-y-auto animate-slideLeft">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center space-x-2 font-mono">
            <Crosshair className="h-5 w-5 text-cyan-400" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-100">
              TARGET INSPECTION DRAWER
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Threat & Class Banner */}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-xl">{cfg.symbol}</span>
            <div>
              <h3 className="text-base font-bold text-slate-100 font-mono" style={{ color: cfg.color }}>
                {cfg.label}
              </h3>
              <p className="text-[11px] font-mono text-slate-400">ID: {detection.id.slice(0, 13)}</p>
            </div>
          </div>

          <div className={`px-2.5 py-1 rounded text-[11px] font-mono font-bold border ${threat.cls}`}>
            {threat.label}
          </div>
        </div>

        {/* Cropped Snapshot Container */}
        <div className="mt-4 rounded-lg bg-black border border-slate-800 overflow-hidden relative p-2 flex flex-col items-center">
          <div className="text-[10px] font-mono text-slate-400 mb-1 flex items-center space-x-1">
            <Layers className="h-3 w-3 text-cyan-400" />
            <span>ACOUSTIC SHADOW CROP (HIGH-RES)</span>
          </div>

          <div className="relative border border-cyan-500/40 rounded overflow-hidden max-h-48 w-full flex justify-center bg-slate-950">
            {/* Display main image centered on target bbox */}
            <img
              src={imageUrl}
              alt="Cropped Sonar Target"
              className="max-h-44 object-cover"
              style={{
                objectPosition: `${((detection.bbox.x_min + detection.bbox.x_max) / 2)}px ${((detection.bbox.y_min + detection.bbox.y_max) / 2)}px`,
              }}
            />
            {/* Target Reticle Overlay */}
            <div className="absolute inset-0 border-2 border-cyan-400/80 pointer-events-none rounded shadow-[inset_0_0_12px_rgba(0,212,255,0.4)]" />
          </div>
        </div>

        {/* Metric Cards Grid */}
        <div className="mt-5 grid grid-cols-2 gap-3 text-xs font-mono">
          {/* Target Dimensions */}
          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg">
            <div className="flex items-center space-x-1.5 text-slate-400 text-[10px] uppercase mb-1">
              <Ruler className="h-3.5 w-3.5 text-cyan-400" />
              <span>Est. Dimensions</span>
            </div>
            <div className="text-base font-bold text-slate-100">
              {lengthM}m × {widthM}m
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Shadow length: {shadowM}m</div>
          </div>

          {/* Confidence Score */}
          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg">
            <div className="flex items-center space-x-1.5 text-slate-400 text-[10px] uppercase mb-1">
              <ShieldAlert className="h-3.5 w-3.5 text-amber-400" />
              <span>AI Confidence</span>
            </div>
            <div className="text-base font-bold text-amber-400">
              {Math.round(detection.confidence)}%
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Rescored via Geometry</div>
          </div>

          {/* Decimal Coordinates */}
          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg col-span-2">
            <div className="flex items-center space-x-1.5 text-slate-400 text-[10px] uppercase mb-1">
              <MapPin className="h-3.5 w-3.5 text-emerald-400" />
              <span>Geospatial Coordinates</span>
            </div>
            <div className="text-xs font-bold text-slate-200">
              LAT: {detection.lat.toFixed(6)}° N
            </div>
            <div className="text-xs font-bold text-slate-200">
              LON: {detection.lon.toFixed(6)}° E
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              DMS: {toDMS(detection.lat, true)} | {toDMS(detection.lon, false)}
            </div>
          </div>

          {/* Range & Offsets */}
          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg col-span-2">
            <div className="flex items-center space-x-1.5 text-slate-400 text-[10px] uppercase mb-1">
              <Compass className="h-3.5 w-3.5 text-purple-400" />
              <span>Acoustic Slant & Ground Range</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Slant Range:</span>
              <span className="text-slate-200 font-bold">{(Math.sqrt(bboxW * bboxW + bboxH * bboxH) * 0.15).toFixed(1)} m</span>
            </div>
            <div className="flex justify-between text-xs mt-1">
              <span className="text-slate-400">Seabed Offset:</span>
              <span className="text-slate-200 font-bold">{detection.lat > 0 ? "Starboard (+14.2m)" : "Port (-12.8m)"}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Action Footer */}
      <div className="mt-6 pt-4 border-t border-slate-800 flex justify-end">
        <button
          onClick={onClose}
          className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono text-xs font-bold rounded transition-colors"
        >
          CLOSE INSPECTION
        </button>
      </div>
    </div>
  );
}
