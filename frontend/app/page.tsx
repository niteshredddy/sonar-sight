"use client";

import { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { uploadSonarImage, processJob, getJobStatus } from "@/lib/api";
import { CLASS_CONFIG } from "@/lib/types";
import { Radio, UploadCloud, FileText, CheckCircle2, ShieldAlert, Sliders, ArrowRight } from "lucide-react";

export default function UploadPage() {
  const router = useRouter();
  const [sonarFile, setSonarFile] = useState<File | null>(null);
  const [metadataFile, setMetadataFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const [progress, setProgress] = useState<number>(0);
  const [error, setError] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const sonarInputRef = useRef<HTMLInputElement>(null);
  const metaInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find((f) =>
      /\.(png|jpg|jpeg|tif|tiff|bmp)$/i.test(f.name)
    );
    const metaFile = files.find((f) => /\.(json|csv)$/i.test(f.name));
    if (imageFile) setSonarFile(imageFile);
    if (metaFile) setMetadataFile(metaFile);
  }, []);

  const pollStatus = async (jobId: string) => {
    const maxAttempts = 120;
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      try {
        const s = await getJobStatus(jobId);
        setStatus(s.message || s.status);
        setProgress(s.progress || 0);

        if (s.status === "done") {
          return true;
        }
        if (s.status === "error") {
          setError(s.message || "Processing failed");
          return false;
        }
      } catch {
        // Retry on transient errors
      }
    }
    setError("Processing timed out");
    return false;
  };

  const handleSubmit = async () => {
    if (!sonarFile) return;
    setIsProcessing(true);
    setError("");
    setStatus("Uploading survey tile...");
    setProgress(15);

    try {
      const { job_id } = await uploadSonarImage(sonarFile, metadataFile);
      setStatus("Initializing detection pipeline...");
      setProgress(30);

      await processJob(job_id);
      setStatus("Filtering & Geotagging...");

      const success = await pollStatus(job_id);
      if (success) {
        router.push(`/results/${job_id}`);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#070c14] text-slate-200 flex flex-col font-sans">
      {/* Top Header */}
      <header className="w-full glass-panel border-b border-cyan-500/20 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-cyan-950 border border-cyan-400/50 shadow-[0_0_12px_rgba(0,212,255,0.3)]">
            <Radio className="h-6 w-6 text-cyan-400 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-wider uppercase text-slate-100 font-mono">
              SONARSIGHT <span className="text-cyan-400 text-xs font-normal">v2.4 COMMAND</span>
            </h1>
            <p className="text-xs font-mono text-slate-400">SIDE-SCAN SONAR MARINE DEBRIS INFERENCE CONSOLE</p>
          </div>
        </div>
      </header>

      {/* Main Upload Portal */}
      <main className="flex-1 max-w-4xl w-full mx-auto p-6 space-y-6 flex flex-col justify-center">
        {/* Title Card */}
        <div className="text-center space-y-2">
          <span className="px-3 py-1 rounded text-xs font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-500/40 uppercase tracking-widest">
            SIH DEFENSE-GRADE DEPLOYMENT
          </span>
          <h2 className="text-3xl font-black uppercase tracking-tight text-white font-mono">
            TACTICAL MISSION SURVEY UPLOAD
          </h2>
          <p className="text-sm font-mono text-slate-400 max-w-xl mx-auto">
            Drag and drop side-scan sonar waterfall strips and per-ping navigation metadata for real-time AI detection, nadir masking, and slant-range georeferencing.
          </p>
        </div>

        {/* Upload Card */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`glass-panel p-8 rounded-xl border-2 transition-all text-center space-y-6 ${
            isDragging
              ? "border-cyan-400 bg-cyan-950/40 shadow-[0_0_30px_rgba(0,212,255,0.2)]"
              : sonarFile
              ? "border-emerald-500/50 bg-emerald-950/10"
              : "border-cyan-500/20 hover:border-cyan-500/50"
          }`}
        >
          <input
            ref={sonarInputRef}
            type="file"
            accept="image/*,.tif,.tiff"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && setSonarFile(e.target.files[0])}
          />
          <input
            ref={metaInputRef}
            type="file"
            accept=".json,.csv"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && setMetadataFile(e.target.files[0])}
          />

          <div className="flex flex-col items-center space-y-3">
            <div className="h-16 w-16 rounded-full bg-slate-900 border border-cyan-500/30 flex items-center justify-center shadow-lg">
              <UploadCloud className="h-8 w-8 text-cyan-400" />
            </div>
            <div>
              <p className="text-base font-bold font-mono text-slate-100 uppercase">
                {sonarFile ? sonarFile.name : "DROP SONAR WATERFALL IMAGE HERE"}
              </p>
              <p className="text-xs font-mono text-slate-400 mt-1">
                Supports PNG, JPG, TIFF side-scan sonar image tiles
              </p>
            </div>
            <button
              type="button"
              onClick={() => sonarInputRef.current?.click()}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded font-mono text-xs font-bold border border-slate-700 transition-colors cursor-pointer"
            >
              Select Sonar File
            </button>
          </div>

          {/* Metadata Section */}
          <div className="pt-4 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-4 font-mono text-xs text-slate-400">
            <div className="flex items-center space-x-2">
              <FileText className="h-4 w-4 text-amber-400" />
              <span>
                Metadata Track: <strong className="text-slate-200">{metadataFile ? metadataFile.name : "Auto-generated (Offshore Arabian Sea)"}</strong>
              </span>
            </div>
            <button
              type="button"
              onClick={() => metaInputRef.current?.click()}
              className="px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-800 text-[11px] cursor-pointer"
            >
              {metadataFile ? "Change Metadata File" : "Attach Navigation JSON/CSV"}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-4 rounded-lg bg-red-950/80 border border-red-500/50 text-red-400 font-mono text-xs flex items-center space-x-2">
            <ShieldAlert className="h-5 w-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Progress Bar */}
        {isProcessing && (
          <div className="glass-panel p-4 rounded-lg space-y-2 font-mono text-xs">
            <div className="flex justify-between text-slate-300">
              <span className="text-cyan-400 font-bold uppercase">{status || "PROCESSING..."}</span>
              <span className="font-bold">{progress}%</span>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
              <div
                className="bg-cyan-400 h-full transition-all duration-300 shadow-[0_0_12px_#00d4ff]"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Target Classes Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
          {Object.entries(CLASS_CONFIG).map(([key, cfg]) => (
            <div key={key} className="glass-panel p-3 rounded-lg border border-slate-800 text-center">
              <div className="text-lg mb-1">{cfg.symbol}</div>
              <div className="font-bold uppercase" style={{ color: cfg.color }}>{cfg.label}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Target AI Class</div>
            </div>
          ))}
        </div>

        {/* Submit Action */}
        <div className="pt-2">
          <button
            onClick={handleSubmit}
            disabled={!sonarFile || isProcessing}
            className="w-full py-3.5 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 text-slate-950 font-black font-mono text-sm uppercase tracking-wider rounded-lg transition-all shadow-[0_0_20px_rgba(0,212,255,0.4)] flex items-center justify-center space-x-2 cursor-pointer"
          >
            <span>{isProcessing ? "PROCESSING MISSION..." : "LAUNCH DETECTION PIPELINE"}</span>
            <ArrowRight className="h-5 w-5" />
          </button>
        </div>
      </main>
    </div>
  );
}
