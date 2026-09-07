"use client";

import { useEffect, useRef, useState } from "react";
import type { Detection, ClassLabel } from "@/lib/types";
import { CLASS_CONFIG } from "@/lib/types";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Layers, Eye, EyeOff } from "lucide-react";

interface MapComponentProps {
  detections: Detection[];
  selectedId: string | null;
  hoveredId: string | null;
  onSelectDetection: (id: string | null) => void;
  visibleClasses: Record<ClassLabel, boolean>;
}

export default function MapComponent({
  detections,
  selectedId,
  hoveredId,
  onSelectDetection,
  visibleClasses,
}: MapComponentProps) {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const markersRef = useRef<Map<string, L.Marker>>(new Map());
  const seaMarkLayerRef = useRef<L.TileLayer | null>(null);
  const [showBathymetry, setShowBathymetry] = useState(true);

  // Filter visible detections
  const activeDetections = detections.filter((d) => visibleClasses[d.class_label] !== false);

  // Initialize Map
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    // Default offshore Arabian Sea (Mumbai offshore waters)
    const map = L.map(containerRef.current, {
      center: [18.925, 72.68],
      zoom: 13,
      zoomControl: false,
    });

    L.control.zoom({ position: "topright" }).addTo(map);

    // 1. Dark Basemap (OSM tiles + CSS dark filter, no API key required)
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(map);

    // 2. OpenSeaMap Bathymetry / Nautical Seamark Overlay
    const seaMarkLayer = L.tileLayer("https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png", {
      attribution: 'Map data &copy; <a href="http://www.openseamap.org">OpenSeaMap</a> contributors',
      maxZoom: 18,
      opacity: 0.8,
    }).addTo(map);

    seaMarkLayerRef.current = seaMarkLayer;
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Toggle Bathymetry Overlay
  useEffect(() => {
    if (!mapRef.current || !seaMarkLayerRef.current) return;
    if (showBathymetry) {
      seaMarkLayerRef.current.addTo(mapRef.current);
    } else {
      seaMarkLayerRef.current.remove();
    }
  }, [showBathymetry]);

  // Update Pulsing Radar Ping Markers
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Clear old markers
    markersRef.current.forEach((marker) => marker.remove());
    markersRef.current.clear();

    const bounds = L.latLngBounds([]);

    activeDetections.forEach((det) => {
      const isSelected = det.id === selectedId;
      const isHovered = det.id === hoveredId;

      // Color mapping for radar ping
      let pingClass = "radar-ping-cyan";
      if (det.class_label === "ghost_net") pingClass = "radar-ping-amber";
      if (det.class_label === "shipwreck") pingClass = "radar-ping-crimson";
      if (det.class_label === "cylinder") pingClass = "radar-ping-emerald";

      const cfg = CLASS_CONFIG[det.class_label] || CLASS_CONFIG.ghost_net;

      // Tactical Pulse Ping DivIcon
      const icon = L.divIcon({
        className: "custom-radar-marker",
        html: `
          <div class="relative flex items-center justify-center w-6 h-6">
            <div class="w-4 h-4 rounded-full ${pingClass} ${
          isSelected ? "scale-125 border-2 border-white" : ""
        }"></div>
            ${
              isSelected
                ? `<div class="absolute -inset-1 rounded-full border border-cyan-400 animate-ping"></div>`
                : ""
            }
          </div>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      const marker = L.marker([det.lat, det.lon], { icon }).addTo(map);

      // Popup content
      const popupContent = `
        <div style="font-family: monospace; font-size: 11px; color: #f8fafc;">
          <div style="font-weight: bold; color: ${cfg.color}; font-size: 12px;">${cfg.label.toUpperCase()}</div>
          <div>Confidence: <b>${Math.round(det.confidence)}%</b></div>
          <div>Lat: ${det.lat.toFixed(6)}°</div>
          <div>Lon: ${det.lon.toFixed(6)}°</div>
        </div>
      `;

      marker.bindPopup(popupContent);
      marker.on("click", () => onSelectDetection(det.id));

      markersRef.current.set(det.id, marker);
      bounds.extend([det.lat, det.lon]);
    });

    // Auto-fit bounds if detections exist
    if (activeDetections.length > 0 && bounds.isValid()) {
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 15 });
    }
  }, [activeDetections, selectedId, hoveredId]);

  // Pan to selected detection
  useEffect(() => {
    if (!selectedId || !mapRef.current) return;
    const det = activeDetections.find((d) => d.id === selectedId);
    if (det) {
      mapRef.current.panTo([det.lat, det.lon], { animate: true, duration: 0.5 });
      const marker = markersRef.current.get(selectedId);
      if (marker) marker.openPopup();
    }
  }, [selectedId]);

  return (
    <div className="relative w-full h-full glass-panel rounded-lg overflow-hidden border border-cyan-500/20">
      {/* Top Map Bar */}
      <div className="absolute top-3 left-3 z-[1000] flex items-center space-x-2 font-mono text-xs">
        <button
          onClick={() => setShowBathymetry(!showBathymetry)}
          className={`px-3 py-1.5 rounded flex items-center space-x-1.5 border transition-all cursor-pointer ${
            showBathymetry
              ? "bg-cyan-950/90 text-cyan-300 border-cyan-400 shadow-[0_0_10px_rgba(0,212,255,0.3)]"
              : "bg-slate-900/90 text-slate-400 border-slate-700 hover:text-slate-200"
          }`}
        >
          <Layers className="h-3.5 w-3.5" />
          <span>Bathymetry Overlay</span>
          {showBathymetry ? <Eye className="h-3 w-3 text-cyan-400 ml-1" /> : <EyeOff className="h-3 w-3 text-slate-500 ml-1" />}
        </button>
      </div>

      <div ref={containerRef} className="w-full h-full min-h-[380px]" />
    </div>
  );
}
