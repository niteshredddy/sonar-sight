"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { getReport, getUploadUrl, processJob } from "@/lib/api";
import type {
  Detection,
  DetectionReport,
  ClassLabel,
  AcousticPalette,
  ProcessConfig,
  TelemetryData,
} from "@/lib/types";
import { CLASS_CONFIG } from "@/lib/types";
import dynamic from "next/dynamic";
import HUDHeader from "@/app/components/HUDHeader";
import SonarViewport from "@/app/components/SonarViewport";
import TargetInspector from "@/app/components/TargetInspector";
import {
  ArrowUpDown,
  Search,
  Filter,
  RefreshCw,
  Sliders,
  Eye,
  Crosshair,
  MapPin,
  ShieldAlert,
} from "lucide-react";

// Dynamic import for Leaflet map component (SSR incompatible)
const MapComponent = dynamic(() => import("./MapComponent"), { ssr: false });

type SortKey = "confidence" | "class_label" | "lat" | "lon" | "slant_range";

export default function ResultsPage() {
  const params = useParams();
  const jobId = params.jobId as string;

  const [report, setReport] = useState<DetectionReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("confidence");
  const [sortAsc, setSortAsc] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  // Dynamic process configuration state
  const [config, setConfig] = useState<ProcessConfig>({
    conf_threshold: 0.45,
    iou_threshold: 0.45,
    min_area_pixels: 100,
    enable_nadir_mask: true,
    palette: "amber",
  });

  // Class visibility filters
  const [visibleClasses, setVisibleClasses] = useState<Record<ClassLabel, boolean>>({
    ghost_net: true,
    shipwreck: true,
    pipe: true,
    cylinder: true,
  });

  const fetchReportData = useCallback(async () => {
    try {
      setLoading(true);
      const r = await getReport(jobId);
      setReport(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load report");
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchReportData();
  }, [fetchReportData]);

  // Handle Dynamic Re-filtering trigger from HUD Sliders
  const handleApplyConfig = async () => {
    try {
      setIsProcessing(true);
      await processJob(jobId, {
        conf_threshold: config.conf_threshold,
        iou_threshold: config.iou_threshold,
        min_area_pixels: config.min_area_pixels,
        enable_nadir_mask: config.enable_nadir_mask,
      });

      // Poll until done
      let attempts = 0;
      const interval = setInterval(async () => {
        attempts++;
        const r = await getReport(jobId);
        if (r && r.detections) {
          setReport(r);
          setIsProcessing(false);
          clearInterval(interval);
        } else if (attempts > 15) {
          setIsProcessing(false);
          clearInterval(interval);
        }
      }, 1000);
    } catch (e) {
      alert("Error re-filtering pipeline: " + e);
      setIsProcessing(false);
    }
  };

  const handleConfigChange = (newConfig: Partial<ProcessConfig>) => {
    setConfig((prev) => ({ ...prev, ...newConfig }));
  };

  const handleToggleClass = (cls: ClassLabel) => {
    setVisibleClasses((prev) => ({ ...prev, [cls]: !prev[cls] }));
  };

  // Filter & Sort Detections for Table
  const activeDetections = (report?.detections || []).filter(
    (d) => visibleClasses[d.class_label] !== false
  );

  const sortedDetections = [...activeDetections].sort((a, b) => {
    const mul = sortAsc ? 1 : -1;
    if (sortKey === "confidence") return mul * (a.confidence - b.confidence);
    if (sortKey === "class_label") return mul * a.class_label.localeCompare(b.class_label);
    if (sortKey === "lat") return mul * (a.lat - b.lat);
    if (sortKey === "lon") return mul * (a.lon - b.lon);
    return 0;
  });

  // Calculate live mission telemetry
  const criticalCount = activeDetections.filter((d) => d.confidence >= 80).length;
  const highCount = activeDetections.filter((d) => d.confidence >= 60 && d.confidence < 80).length;

  const threatScore =
    criticalCount > 0 ? "CRITICAL" : highCount > 0 ? "HIGH" : activeDetections.length > 0 ? "MEDIUM" : "LOW";

  const telemetry: TelemetryData = {
    vessel_sog_kts: 3.4,
    towfish_altitude_m: 12.5,
    swath_width_m: 100.0,
    water_depth_m: 42.0,
    active_targets: activeDetections.length,
    threat_score: threatScore,
  };

  const selectedDetection = activeDetections.find((d) => d.id === selectedId) || null;
  const imageUrl = report ? getUploadUrl(jobId, report.source_file) : "";

  if (loading && !report) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-[#070c14] text-cyan-400 font-mono">
        <div className="flex flex-col items-center space-y-3">
          <RefreshCw className="h-8 w-8 animate-spin text-cyan-400" />
          <p className="text-sm font-bold uppercase tracking-widest">
            INITIALIZING TACTICAL CONSOLE & REPORT...
          </p>
        </div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-[#070c14] text-red-400 font-mono p-4">
        <div className="glass-panel p-6 rounded-lg max-w-md border border-red-500/30 text-center">
          <ShieldAlert className="h-10 w-10 text-red-500 mx-auto mb-3" />
          <h2 className="text-base font-bold uppercase mb-2">SURVEY MISSION ERROR</h2>
          <p className="text-xs text-slate-300 mb-4">{error || "Could not retrieve mission report."}</p>
          <a
            href="/"
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded font-bold text-xs inline-block"
          >
            RETURN TO MISSION UPLOAD
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#070c14] text-slate-200 flex flex-col font-sans selection:bg-cyan-500/30">
      {/* Top Telemetry & HUD Banner */}
      <HUDHeader
        jobId={jobId}
        telemetry={telemetry}
        config={config}
        onConfigChange={handleConfigChange}
        onApplyConfig={handleApplyConfig}
        isProcessing={isProcessing}
      />

      {/* Main Command Workspace */}
      <main className="flex-1 p-3 space-y-3 overflow-y-auto">
        {/* Workspace Split Grid: Left Sonar Viewport | Right Nautical Map */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 h-[520px]">
          {/* Left Column: Interactive Sonar Waterfall Viewport (7 Cols) */}
          <div className="lg:col-span-7 h-full">
            <SonarViewport
              imageUrl={imageUrl}
              detections={report.detections}
              selectedId={selectedId}
              hoveredId={hoveredId}
              onSelectDetection={(id) => setSelectedId(id)}
              onHoverDetection={(id) => setHoveredId(id)}
              palette={config.palette}
              onPaletteChange={(p) => handleConfigChange({ palette: p })}
              visibleClasses={visibleClasses}
              onToggleClass={handleToggleClass}
            />
          </div>

          {/* Right Column: Dark Nautical Map & Bathymetry (5 Cols) */}
          <div className="lg:col-span-5 h-full">
            <MapComponent
              detections={report.detections}
              selectedId={selectedId}
              hoveredId={hoveredId}
              onSelectDetection={(id) => setSelectedId(id)}
              visibleClasses={visibleClasses}
            />
          </div>
        </div>

        {/* Telemetry Debris Data Log Table */}
        <div className="glass-panel rounded-lg overflow-hidden border border-cyan-500/20">
          <div className="px-4 py-2.5 bg-slate-950/90 border-b border-slate-800 flex items-center justify-between font-mono text-xs">
            <div className="flex items-center space-x-2">
              <Crosshair className="h-4 w-4 text-cyan-400" />
              <span className="font-bold text-slate-100 uppercase tracking-wider">
                TACTICAL DEBRIS LOG ({sortedDetections.length} TARGETS)
              </span>
            </div>
            <span className="text-[11px] text-slate-400">
              CLICK ANY ROW TO OPEN TARGET INSPECTION DRAWER
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse font-mono text-xs">
              <thead>
                <tr className="bg-slate-900/80 text-slate-400 border-b border-slate-800 uppercase text-[10px]">
                  <th className="py-2.5 px-4">Target Class</th>
                  <th className="py-2.5 px-4 cursor-pointer" onClick={() => { setSortKey("confidence"); setSortAsc(!sortAsc); }}>
                    <div className="flex items-center space-x-1">
                      <span>Confidence</span>
                      <ArrowUpDown className="h-3 w-3 text-cyan-400" />
                    </div>
                  </th>
                  <th className="py-2.5 px-4 cursor-pointer" onClick={() => { setSortKey("lat"); setSortAsc(!sortAsc); }}>
                    <div className="flex items-center space-x-1">
                      <span>Latitude</span>
                      <ArrowUpDown className="h-3 w-3 text-cyan-400" />
                    </div>
                  </th>
                  <th className="py-2.5 px-4 cursor-pointer" onClick={() => { setSortKey("lon"); setSortAsc(!sortAsc); }}>
                    <div className="flex items-center space-x-1">
                      <span>Longitude</span>
                      <ArrowUpDown className="h-3 w-3 text-cyan-400" />
                    </div>
                  </th>
                  <th className="py-2.5 px-4">Est. BBox (WxH)</th>
                  <th className="py-2.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {sortedDetections.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="text-center py-6 text-slate-500">
                      NO DEBRIS TARGETS DETECTED AT CURRENT THRESHOLDS
                    </td>
                  </tr>
                ) : (
                  sortedDetections.map((det) => {
                    const isSelected = det.id === selectedId;
                    const isHovered = det.id === hoveredId;
                    const cfg = CLASS_CONFIG[det.class_label] || CLASS_CONFIG.ghost_net;

                    const bboxW = Math.round(Math.abs(det.bbox.x_max - det.bbox.x_min));
                    const bboxH = Math.round(Math.abs(det.bbox.y_max - det.bbox.y_min));

                    return (
                      <tr
                        key={det.id}
                        onClick={() => setSelectedId(det.id)}
                        onMouseEnter={() => setHoveredId(det.id)}
                        onMouseLeave={() => setHoveredId(null)}
                        className={`transition-colors cursor-pointer ${
                          isSelected
                            ? "bg-cyan-950/60 text-white font-bold border-l-4 border-cyan-400"
                            : isHovered
                            ? "bg-slate-800/40 text-slate-100"
                            : "hover:bg-slate-900/40 text-slate-300"
                        }`}
                      >
                        <td className="py-2.5 px-4 font-semibold flex items-center space-x-2">
                          <span style={{ color: cfg.color }}>{cfg.symbol}</span>
                          <span style={{ color: cfg.color }}>{cfg.label}</span>
                        </td>
                        <td className="py-2.5 px-4">
                          <span
                            className={`px-2 py-0.5 rounded font-bold ${
                              det.confidence >= 80
                                ? "bg-emerald-950 text-emerald-400 border border-emerald-500/30"
                                : det.confidence >= 50
                                ? "bg-amber-950 text-amber-400 border border-amber-500/30"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {Math.round(det.confidence)}%
                          </span>
                        </td>
                        <td className="py-2.5 px-4 text-slate-300">{det.lat.toFixed(6)}° N</td>
                        <td className="py-2.5 px-4 text-slate-300">{det.lon.toFixed(6)}° E</td>
                        <td className="py-2.5 px-4 text-slate-400">
                          {bboxW} × {bboxH} px
                        </td>
                        <td className="py-2.5 px-4 text-right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedId(det.id);
                            }}
                            className="px-2 py-1 rounded bg-slate-800 hover:bg-cyan-950 hover:text-cyan-300 border border-slate-700 text-[10px] uppercase font-bold transition-all"
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Target Inspection Drawer */}
      {selectedDetection && (
        <TargetInspector
          detection={selectedDetection}
          onClose={() => setSelectedId(null)}
          imageUrl={imageUrl}
        />
      )}
    </div>
  );
}
