"use client";

import { useState } from "react";
import { Box, Download, Eye, Orbit, Sparkles, Triangle } from "lucide-react";
import GaussianViewer from "@/components/GaussianViewer";
import MultiModeSceneViewer from "@/components/MultiModeSceneViewer";
import { getExportUrl } from "@/lib/api";
import type { JobResponse, PreviewRenderMode } from "@/lib/types";

interface LivePreviewPanelProps {
  job: JobResponse | null;
  sceneUrl: string | null;
  gaussianFormat?: "splat" | "ply";
  revision: number;
  heightClass?: string;
  label?: string;
}

const MODE_OPTIONS: {
  id: PreviewRenderMode;
  title: string;
  subtitle: string;
  icon: typeof Sparkles;
  accent: string;
}[] = [
  {
    id: "gaussian",
    title: "Gaussian Splatting",
    subtitle: "Photorealistic anisotropic 3DGS splats",
    icon: Sparkles,
    accent: "text-sky-400 bg-sky-500/10 ring-sky-500/30",
  },
  {
    id: "geometry",
    title: "Geometry",
    subtitle: "3D points, ellipsoid axes & camera poses",
    icon: Orbit,
    accent: "text-emerald-400 bg-emerald-500/10 ring-emerald-500/30",
  },
  {
    id: "mesh",
    title: "Triangle Mesh",
    subtitle: "Marching Cubes surface from Gaussian density",
    icon: Box,
    accent: "text-violet-400 bg-violet-500/10 ring-violet-500/30",
  },
  {
    id: "triangle_splats",
    title: "Triangle Splatting",
    subtitle: "Oriented 3D tangent-plane triangle surfels",
    icon: Triangle,
    accent: "text-amber-400 bg-amber-500/10 ring-amber-500/30",
  },
];

const FOV_PRESETS = [40, 60, 75, 90, 110];

export default function LivePreviewPanel({
  job,
  sceneUrl,
  gaussianFormat = "splat",
  revision,
  heightClass = "h-[480px]",
  label,
}: LivePreviewPanelProps) {
  const [mode, setMode] = useState<PreviewRenderMode>("gaussian");
  const [modeStats, setModeStats] = useState<string>("");
  const [fov, setFov] = useState<number>(65);

  const eqFocalMm = Math.round(18 / Math.tan(((Math.max(25, Math.min(120, fov)) / 2) * Math.PI) / 180));

  const hasAnyData = Boolean(
    job &&
      (job.has_preview ||
        job.has_splat ||
        job.has_ply ||
        job.preview_batches.length > 0 ||
        job.num_gaussians > 0 ||
        job.status === "completed")
  );

  return (
    <div className="grid gap-4 lg:grid-cols-[270px_1fr]">
      {/* Representation & FOV Sidebar */}
      <aside className="flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Render Representation
          </h3>
          <div className="mt-3 space-y-2">
            {MODE_OPTIONS.map((opt) => {
              const active = mode === opt.id;
              const Icon = opt.icon;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => {
                    setMode(opt.id);
                    setModeStats("");
                  }}
                  className={`flex w-full items-start gap-3 rounded-lg border p-3 text-left transition ${
                    active
                      ? "border-sky-500/60 bg-slate-800/90 shadow-sm"
                      : "border-slate-800 bg-slate-950/50 hover:border-slate-700 hover:bg-slate-900"
                  }`}
                >
                  <span className={`mt-0.5 inline-flex rounded-md p-1.5 ring-1 ${opt.accent}`}>
                    <Icon className="h-4 w-4" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-semibold text-white">{opt.title}</span>
                      {active && <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />}
                    </div>
                    <p className="mt-0.5 text-[11px] leading-snug text-slate-400">{opt.subtitle}</p>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Interactive Field-of-View Controls */}
          <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <div className="flex items-center justify-between">
              <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-200">
                <Eye className="h-3.5 w-3.5 text-sky-400" />
                <span>Field of View</span>
              </span>
              <span className="font-mono text-[11px] text-sky-300">~{eqFocalMm}mm eq.</span>
            </div>

            <div className="mt-2 flex items-center gap-2">
              <input
                type="range"
                min={25}
                max={120}
                step={1}
                value={fov}
                onChange={(e) => setFov(Number(e.target.value))}
                className="flex-1"
              />
              <input
                type="number"
                min={25}
                max={120}
                value={Math.round(fov)}
                onChange={(e) => setFov(Math.max(25, Math.min(120, Number(e.target.value) || 65)))}
                className="w-12 rounded border border-slate-700 bg-slate-900 px-1 py-0.5 text-center font-mono text-xs text-white"
              />
              <span className="text-xs text-slate-400">°</span>
            </div>

            <div className="mt-2 flex flex-wrap gap-1">
              {FOV_PRESETS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setFov(preset)}
                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    Math.round(fov) === preset
                      ? "bg-sky-500 text-white"
                      : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                  }`}
                >
                  {preset}°
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-4 space-y-2 border-t border-slate-800 pt-3 text-xs text-slate-400">
          <div className="flex justify-between">
            <span>Active Gaussians</span>
            <span className="font-mono text-slate-200">{job ? job.num_gaussians.toLocaleString() : "—"}</span>
          </div>
          <div className="flex justify-between">
            <span>Registered Cameras</span>
            <span className="font-mono text-slate-200">{job ? job.num_cameras : "—"}</span>
          </div>
          {modeStats && (
            <div className="rounded-md bg-slate-950/80 px-2.5 py-1.5 text-[11px] text-sky-300 ring-1 ring-slate-800">
              {modeStats}
            </div>
          )}
          {mode === "mesh" && job && hasAnyData && (
            <a
              href={getExportUrl(job.id, "mesh")}
              download
              className="mt-1 flex w-full items-center justify-center gap-1.5 rounded-lg border border-violet-500/40 bg-violet-500/10 px-3 py-1.5 text-xs font-medium text-violet-200 hover:bg-violet-500/20"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Download Mesh (.OBJ)</span>
            </a>
          )}
        </div>
      </aside>

      {/* Main WebGL Canvas */}
      <div>
        {mode === "gaussian" ? (
          <GaussianViewer
            sceneUrl={sceneUrl}
            format={gaussianFormat}
            revision={revision}
            heightClass={heightClass}
            label={label}
            fov={fov}
            onFovChange={setFov}
          />
        ) : (
          <MultiModeSceneViewer
            jobId={job?.id}
            mode={mode}
            revision={revision}
            hasData={hasAnyData}
            heightClass={heightClass}
            fov={fov}
            onFovChange={setFov}
            label={`${
              mode === "geometry"
                ? "Geometry View"
                : mode === "mesh"
                ? "Marching Cubes Mesh"
                : "Triangle Splatting"
            }${job ? ` · ${job.num_gaussians.toLocaleString()} primitives` : ""}`}
            onStats={setModeStats}
          />
        )}
      </div>
    </div>
  );
}
