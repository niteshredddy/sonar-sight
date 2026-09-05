"use client";

import React, { useState } from "react";
import type { ProcessConfig, TelemetryData } from "@/lib/types";
import { downloadReportCSV, downloadReportGeoJSON } from "@/lib/api";
import {
  Activity,
  Compass,
  Radio,
  Sliders,
  Download,
  ShieldAlert,
  FileCode,
  FileSpreadsheet,
  Eye,
  Crosshair,
  Layers,
} from "lucide-react";

interface HUDHeaderProps {
  jobId: string;
  telemetry: TelemetryData;
  config: ProcessConfig;
  onConfigChange: (newConfig: Partial<ProcessConfig>) => void;
  onApplyConfig: () => void;
  isProcessing: boolean;
}

export default function HUDHeader({
  jobId,
  telemetry,
  config,
  onConfigChange,
  onApplyConfig,
  isProcessing,
}: HUDHeaderProps) {
  const [showSliders, setShowSliders] = useState(false);

  const getThreatColor = (score: string) => {
    switch (score) {
      case "CRITICAL":
        return "text-red-500 bg-red-950/40 border-red-500/40";
      case "HIGH":
        return "text-amber-500 bg-amber-950/40 border-amber-500/40";
      case "MEDIUM":
        return "text-yellow-400 bg-yellow-950/40 border-yellow-400/40";
      default:
        return "text-emerald-400 bg-emerald-950/40 border-emerald-500/40";
    }
  };

  const handleDownloadGeoJSON = async () => {
    try {
      const blob = await downloadReportGeoJSON(jobId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `SonarSight_Mission_${jobId.slice(0, 8)}.geojson`;
      a.click();
    } catch (e) {
      alert("Failed to download GeoJSON: " + e);
    }
  };

  const handleDownloadCSV = async () => {
    try {
      const blob = await downloadReportCSV(jobId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `SonarSight_Survey_${jobId.slice(0, 8)}.csv`;
      a.click();
    } catch (e) {
      alert("Failed to download CSV: " + e);
    }
  };

  return (
    <header className="sticky top-0 z-40 w-full glass-panel border-b border-cyan-500/20 px-4 py-2.5 shadow-2xl">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Left: Branding & Mission Info */}
        <div className="flex items-center space-x-3">
          <div className="relative flex h-9 w-9 items-center justify-center rounded-md bg-cyan-950/80 border border-cyan-400/50 shadow-[0_0_12px_rgba(0,212,255,0.3)]">
            <Radio className="h-5 w-5 text-cyan-400 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-black tracking-wider uppercase text-slate-100 font-mono">
                SONARSIGHT <span className="text-cyan-400 text-xs font-normal">v2.4 TACTICAL</span>
              </h1>
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold bg-cyan-950 text-cyan-400 border border-cyan-500/30">
                LIVE AUV FEED
              </span>
            </div>
            <p className="text-[11px] font-mono text-slate-400">
              MISSION <span className="text-slate-200">{jobId.slice(0, 8)}</span> | ARABIAN SEA OFFSHORE
            </p>
          </div>
        </div>

        {/* Center: Live Telemetry Metric Cards */}
        <div className="hidden lg:flex items-center space-x-3 text-xs font-mono">
          <div className="px-3 py-1.5 rounded bg-slate-900/80 border border-slate-800 flex items-center space-x-2">
            <Compass className="h-4 w-4 text-cyan-400" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase">SOG / Speed</div>
              <div className="font-bold text-slate-200">{telemetry.vessel_sog_kts} kts</div>
            </div>
          </div>

          <div className="px-3 py-1.5 rounded bg-slate-900/80 border border-slate-800 flex items-center space-x-2">
            <Activity className="h-4 w-4 text-emerald-400" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Altitude</div>
              <div className="font-bold text-slate-200">{telemetry.towfish_altitude_m} m</div>
            </div>
          </div>

          <div className="px-3 py-1.5 rounded bg-slate-900/80 border border-slate-800 flex items-center space-x-2">
            <Layers className="h-4 w-4 text-amber-400" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Swath Width</div>
              <div className="font-bold text-slate-200">{telemetry.swath_width_m} m</div>
            </div>
          </div>

          <div className="px-3 py-1.5 rounded bg-slate-900/80 border border-slate-800 flex items-center space-x-2">
            <Crosshair className="h-4 w-4 text-purple-400" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Water Depth</div>
              <div className="font-bold text-slate-200">{telemetry.water_depth_m} m</div>
            </div>
          </div>

          <div className="px-3 py-1.5 rounded bg-slate-900/80 border border-slate-800 flex items-center space-x-2">
            <Eye className="h-4 w-4 text-cyan-400" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Debris Targets</div>
              <div className="font-bold text-cyan-300">{telemetry.active_targets}</div>
            </div>
          </div>

          {/* Threat Rating Badge */}
          <div
            className={`px-3 py-1.5 rounded border flex items-center space-x-1.5 font-bold uppercase tracking-wider ${getThreatColor(
              telemetry.threat_score
            )}`}
          >
            <ShieldAlert className="h-4 w-4" />
            <span>{telemetry.threat_score}</span>
          </div>
        </div>

        {/* Right: Controls & Export Triggers */}
        <div className="flex items-center space-x-2">
          {/* Sliders Toggle Button */}
          <button
            onClick={() => setShowSliders(!showSliders)}
            className={`px-3 py-1.5 rounded text-xs font-mono font-medium flex items-center space-x-1.5 border transition-all ${
              showSliders
                ? "bg-cyan-950 text-cyan-300 border-cyan-400 shadow-[0_0_10px_rgba(0,212,255,0.3)]"
                : "bg-slate-900 text-slate-300 border-slate-700 hover:border-cyan-500/50"
            }`}
          >
            <Sliders className="h-3.5 w-3.5" />
            <span>Thresholds</span>
          </button>

          {/* GeoJSON Shapefile Export */}
          <button
            onClick={handleDownloadGeoJSON}
            className="px-3 py-1.5 rounded text-xs font-mono font-medium bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 hover:border-cyan-400/50 flex items-center space-x-1.5 transition-all cursor-pointer"
            title="Export Hydrographic GeoJSON FeatureCollection"
          >
            <FileCode className="h-3.5 w-3.5 text-cyan-400" />
            <span className="hidden sm:inline">GeoJSON (SHP)</span>
          </button>

          {/* CSV Brief Export */}
          <button
            onClick={handleDownloadCSV}
            className="px-3 py-1.5 rounded text-xs font-mono font-medium bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 hover:border-amber-400/50 flex items-center space-x-1.5 transition-all cursor-pointer"
            title="Export Survey CSV Report"
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-amber-400" />
            <span className="hidden sm:inline">CSV Brief</span>
          </button>
        </div>
      </div>

      {/* Slide-down Interactive Threshold Controls */}
      {showSliders && (
        <div className="mt-3 pt-3 border-t border-cyan-500/20 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono bg-slate-950/70 p-3 rounded-lg border border-slate-800 animate-fadeIn">
          {/* Confidence Slider */}
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>Confidence Gate (Conf):</span>
              <span className="text-cyan-400 font-bold">{Math.round(config.conf_threshold * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={config.conf_threshold}
              onChange={(e) => onConfigChange({ conf_threshold: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* NMS IoU Slider */}
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>NMS Threshold (IoU):</span>
              <span className="text-cyan-400 font-bold">{Math.round(config.iou_threshold * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={config.iou_threshold}
              onChange={(e) => onConfigChange({ iou_threshold: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Min Area Slider */}
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>Min Target Area:</span>
              <span className="text-cyan-400 font-bold">{config.min_area_pixels} px²</span>
            </div>
            <input
              type="range"
              min="10"
              max="1000"
              step="10"
              value={config.min_area_pixels}
              onChange={(e) => onConfigChange({ min_area_pixels: parseInt(e.target.value) })}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Nadir Masking Toggle & Apply Button */}
          <div className="flex items-center justify-between pt-1">
            <label className="flex items-center space-x-2 text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={config.enable_nadir_mask}
                onChange={(e) => onConfigChange({ enable_nadir_mask: e.target.checked })}
                className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0 h-4 w-4"
              />
              <span>Nadir Masking</span>
            </label>

            <button
              onClick={onApplyConfig}
              disabled={isProcessing}
              className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-slate-950 font-bold rounded text-xs transition-all shadow-[0_0_10px_rgba(0,212,255,0.4)] cursor-pointer"
            >
              {isProcessing ? "Filtering..." : "Apply Controls"}
            </button>
          </div>
        </div>
      )}
    </header>
  );
}
