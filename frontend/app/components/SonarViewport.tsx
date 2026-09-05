"use client";

import React, { useRef, useEffect, useState } from "react";
import type { Detection, ClassLabel, AcousticPalette } from "@/lib/types";
import { CLASS_CONFIG } from "@/lib/types";
import { Layers, Eye, Filter, Palette, Maximize2, ZoomIn, ZoomOut, RefreshCw } from "lucide-react";

interface SonarViewportProps {
  imageUrl: string;
  detections: Detection[];
  selectedId: string | null;
  hoveredId: string | null;
  onSelectDetection: (id: string | null) => void;
  onHoverDetection: (id: string | null) => void;
  palette: AcousticPalette;
  onPaletteChange: (p: AcousticPalette) => void;
  visibleClasses: Record<ClassLabel, boolean>;
  onToggleClass: (cls: ClassLabel) => void;
}

export default function SonarViewport({
  imageUrl,
  detections,
  selectedId,
  hoveredId,
  onSelectDetection,
  onHoverDetection,
  palette,
  onPaletteChange,
  visibleClasses,
  onToggleClass,
}: SonarViewportProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState(1);

  // Filter detections based on visible classes
  const activeDetections = detections.filter((d) => visibleClasses[d.class_label] !== false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      // 1. Draw raw image
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // 2. Apply Acoustic Palette False Color Filters
      if (palette !== "grayscale") {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
          // Average luminance
          const avg = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          const norm = avg / 255.0;

          if (palette === "amber") {
            // Warm Sonar Amber Tone
            data[i] = Math.min(255, norm * 255 * 1.1); // R
            data[i + 1] = Math.min(255, Math.pow(norm, 1.2) * 190); // G
            data[i + 2] = Math.min(255, Math.pow(norm, 2.0) * 40); // B
          } else if (palette === "copper") {
            // Thermal Copper Tone
            data[i] = Math.min(255, Math.pow(norm, 0.9) * 255);
            data[i + 1] = Math.min(255, Math.pow(norm, 1.5) * 160);
            data[i + 2] = Math.min(255, Math.pow(norm, 2.5) * 90);
          } else if (palette === "emerald") {
            // Night-Vision Emerald Tone
            data[i] = Math.min(255, Math.pow(norm, 2.0) * 30);
            data[i + 1] = Math.min(255, norm * 255 * 1.1);
            data[i + 2] = Math.min(255, Math.pow(norm, 1.3) * 180);
          }
        }
        ctx.putImageData(imageData, 0, 0);
      }

      // 3. Render Vector HUD Corner Reticles & Leader Lines
      activeDetections.forEach((det, idx) => {
        const isSelected = det.id === selectedId;
        const isHovered = det.id === hoveredId;
        const cfg = CLASS_CONFIG[det.class_label] || CLASS_CONFIG.ghost_net;

        const { x_min, y_min, x_max, y_max } = det.bbox;
        const w = x_max - x_min;
        const h = y_max - y_min;

        // Draw Polygon Segmentation Mask if present
        if (det.mask && det.mask.length > 2) {
          ctx.beginPath();
          ctx.moveTo(det.mask[0][0], det.mask[0][1]);
          for (let p = 1; p < det.mask.length; p++) {
            ctx.lineTo(det.mask[p][0], det.mask[p][1]);
          }
          ctx.closePath();
          ctx.fillStyle = isSelected
            ? "rgba(0, 212, 255, 0.35)"
            : cfg.bg;
          ctx.fill();
        }

        // Vector Reticle Bracket Style
        const strokeColor = isSelected ? "#00d4ff" : isHovered ? "#ffffff" : cfg.color;
        const strokeWidth = isSelected ? 3 : isHovered ? 2.5 : 2;
        const bracketLen = Math.min(w, h, 24) * 0.35;

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = strokeWidth;

        // Top-Left Corner
        ctx.beginPath();
        ctx.moveTo(x_min, y_min + bracketLen);
        ctx.lineTo(x_min, y_min);
        ctx.lineTo(x_min + bracketLen, y_min);
        ctx.stroke();

        // Top-Right Corner
        ctx.beginPath();
        ctx.moveTo(x_max - bracketLen, y_min);
        ctx.lineTo(x_max, y_min);
        ctx.lineTo(x_max, y_min + bracketLen);
        ctx.stroke();

        // Bottom-Left Corner
        ctx.beginPath();
        ctx.moveTo(x_min, y_max - bracketLen);
        ctx.lineTo(x_min, y_max);
        ctx.lineTo(x_min + bracketLen, y_max);
        ctx.stroke();

        // Bottom-Right Corner
        ctx.beginPath();
        ctx.moveTo(x_max - bracketLen, y_max);
        ctx.lineTo(x_max, y_max);
        ctx.lineTo(x_max, y_max - bracketLen);
        ctx.stroke();

        // Glowing center crosshair if selected
        if (isSelected || isHovered) {
          const cx = (x_min + x_max) / 2;
          const cy = (y_min + y_max) / 2;

          ctx.strokeStyle = "#00d4ff";
          ctx.lineWidth = 1;

          ctx.beginPath();
          ctx.moveTo(cx - 8, cy);
          ctx.lineTo(cx + 8, cy);
          ctx.moveTo(cx, cy - 8);
          ctx.lineTo(cx, cy + 8);
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(cx, cy, 10, 0, 2 * Math.PI);
          ctx.stroke();
        }

        // Leader Line & Callout Label Box
        // Alternate diagonal direction based on index to prevent label collisions
        const leaderOffsetX = idx % 2 === 0 ? 35 : -35;
        const leaderOffsetY = -25;

        const tagOriginX = x_max;
        const tagOriginY = y_min;

        const tagEndX = tagOriginX + leaderOffsetX;
        const tagEndY = tagOriginY + leaderOffsetY;

        // Draw leader line (───○)
        ctx.beginPath();
        ctx.moveTo(tagOriginX, tagOriginY);
        ctx.lineTo(tagEndX, tagEndY);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 2]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Small circle anchor at target origin
        ctx.beginPath();
        ctx.arc(tagOriginX, tagOriginY, 3, 0, 2 * Math.PI);
        ctx.fillStyle = strokeColor;
        ctx.fill();

        // Tag label text
        const labelText = `[${cfg.symbol}] ${det.class_label.toUpperCase()} ${Math.round(det.confidence)}%`;
        ctx.font = "bold 11px monospace";
        const textMetrics = ctx.measureText(labelText);
        const textW = textMetrics.width + 10;
        const textH = 18;

        const boxX = leaderOffsetX > 0 ? tagEndX : tagEndX - textW;
        const boxY = tagEndY - textH / 2;

        // Tag background badge
        ctx.fillStyle = isSelected ? "rgba(0, 212, 255, 0.9)" : "rgba(11, 20, 36, 0.9)";
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1;
        ctx.fillRect(boxX, boxY, textW, textH);
        ctx.strokeRect(boxX, boxY, textW, textH);

        // Tag text
        ctx.fillStyle = isSelected ? "#070c14" : "#f8fafc";
        ctx.fillText(labelText, boxX + 5, boxY + 13);
      });
    };
  }, [imageUrl, activeDetections, selectedId, hoveredId, palette]);

  // Click handler to select detection from canvas coordinates
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Find clicked detection
    const hit = activeDetections.find((det) => {
      const { x_min, y_min, x_max, y_max } = det.bbox;
      return clickX >= x_min && clickX <= x_max && clickY >= y_min && clickY <= y_max;
    });

    onSelectDetection(hit ? hit.id : null);
  };

  // Mouse move handler for hover highlight
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    const hit = activeDetections.find((det) => {
      const { x_min, y_min, x_max, y_max } = det.bbox;
      return mouseX >= x_min && mouseX <= x_max && mouseY >= y_min && mouseY <= y_max;
    });

    onHoverDetection(hit ? hit.id : null);
  };

  return (
    <div className="flex flex-col h-full glass-panel rounded-lg overflow-hidden border border-cyan-500/20">
      {/* Control Bar: Palettes & Class Visibility Filters */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-3 py-2 bg-slate-950/90 border-b border-slate-800 text-xs font-mono">
        {/* Acoustic Palette Switcher */}
        <div className="flex items-center space-x-2">
          <Palette className="h-4 w-4 text-cyan-400" />
          <span className="text-slate-400 uppercase text-[11px]">Palette:</span>
          {(["amber", "copper", "grayscale", "emerald"] as AcousticPalette[]).map((p) => (
            <button
              key={p}
              onClick={() => onPaletteChange(p)}
              className={`px-2 py-0.5 rounded capitalize transition-all cursor-pointer ${
                palette === p
                  ? "bg-cyan-950 text-cyan-300 border border-cyan-400 font-bold"
                  : "bg-slate-900 text-slate-400 border border-slate-800 hover:text-slate-200"
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Class Visibility Checkboxes */}
        <div className="flex items-center space-x-3">
          <Filter className="h-4 w-4 text-amber-400" />
          {(["ghost_net", "shipwreck", "pipe", "cylinder"] as ClassLabel[]).map((cls) => {
            const cfg = CLASS_CONFIG[cls];
            const isChecked = visibleClasses[cls] !== false;
            return (
              <label
                key={cls}
                className="flex items-center space-x-1.5 cursor-pointer select-none text-slate-300 hover:text-white"
              >
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => onToggleClass(cls)}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 h-3.5 w-3.5"
                />
                <span style={{ color: cfg.color }} className="font-semibold">
                  {cfg.label}
                </span>
              </label>
            );
          })}
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center space-x-1 text-slate-400">
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            className="p-1 hover:text-cyan-400 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <span className="w-10 text-center text-slate-200">{Math.round(zoom * 100)}%</span>
          <button
            onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}
            className="p-1 hover:text-cyan-400 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            onClick={() => setZoom(1)}
            className="p-1 hover:text-cyan-400 transition-colors"
            title="Reset Zoom"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Main Canvas Sonar Waterfall Display */}
      <div
        ref={containerRef}
        className="relative flex-1 overflow-auto bg-black flex items-center justify-center p-2 min-h-[420px]"
      >
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMouseMove}
          onMouseLeave={() => onHoverDetection(null)}
          style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}
          className="cursor-crosshair transition-transform duration-150 max-w-full h-auto shadow-2xl rounded"
        />
      </div>
    </div>
  );
}
