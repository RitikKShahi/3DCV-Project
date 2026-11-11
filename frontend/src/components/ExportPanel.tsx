"use client";

import { Download, FileCode2, Sparkles } from "lucide-react";
import { getExportUrl } from "@/lib/api";
import type { JobResponse } from "@/lib/types";

export default function ExportPanel({ job }: { job: JobResponse | null }) {
  if (!job) return null;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
        Export Reconstructed 3D Scene
      </h3>
      <p className="mt-1 text-xs text-slate-400">
        Download standard 3D Gaussian Splatting formats compatible with SuperSplat, Three.js, and WebGL viewers.
      </p>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <a
          href={job.has_ply ? getExportUrl(job.id, "ply") : undefined}
          download
          aria-disabled={!job.has_ply}
          className={`flex items-center justify-between rounded-lg border px-4 py-3 text-sm font-medium transition ${
            job.has_ply
              ? "border-sky-500/40 bg-sky-500/10 text-sky-200 hover:bg-sky-500/20"
              : "pointer-events-none border-slate-800 bg-slate-900/40 text-slate-500"
          }`}
        >
          <div>
            <div className="flex items-center gap-1.5">
              <Sparkles className="h-4 w-4" />
              <span>Standard .PLY</span>
            </div>
            <p className="mt-0.5 text-[11px] opacity-75">Full 3DGS attributes + SH</p>
          </div>
          <Download className="h-4 w-4" />
        </a>

        <a
          href={job.has_splat ? getExportUrl(job.id, "splat") : undefined}
          download
          aria-disabled={!job.has_splat}
          className={`flex items-center justify-between rounded-lg border px-4 py-3 text-sm font-medium transition ${
            job.has_splat
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20"
              : "pointer-events-none border-slate-800 bg-slate-900/40 text-slate-500"
          }`}
        >
          <div>
            <div className="flex items-center gap-1.5">
              <Sparkles className="h-4 w-4" />
              <span>Compact .SPLAT</span>
            </div>
            <p className="mt-0.5 text-[11px] opacity-75">32B/splat WebGL binary</p>
          </div>
          <Download className="h-4 w-4" />
        </a>

        <a
          href={job.has_ply || job.has_preview ? getExportUrl(job.id, "mesh") : undefined}
          download
          aria-disabled={!job.has_ply && !job.has_preview}
          className={`flex items-center justify-between rounded-lg border px-4 py-3 text-sm font-medium transition ${
            job.has_ply || job.has_preview
              ? "border-violet-500/40 bg-violet-500/10 text-violet-200 hover:bg-violet-500/20"
              : "pointer-events-none border-slate-800 bg-slate-900/40 text-slate-500"
          }`}
        >
          <div>
            <div className="flex items-center gap-1.5">
              <FileCode2 className="h-4 w-4" />
              <span>Mesh .OBJ</span>
            </div>
            <p className="mt-0.5 text-[11px] opacity-75">Marching Cubes surface</p>
          </div>
          <Download className="h-4 w-4" />
        </a>

        <a
          href={job.status === "completed" ? getExportUrl(job.id, "benchmarks") : undefined}
          download
          aria-disabled={job.status !== "completed"}
          className={`flex items-center justify-between rounded-lg border px-4 py-3 text-sm font-medium transition ${
            job.status === "completed"
              ? "border-slate-700 bg-slate-800/70 text-slate-200 hover:bg-slate-800"
              : "pointer-events-none border-slate-800 bg-slate-900/40 text-slate-500"
          }`}
        >
          <div>
            <div className="flex items-center gap-1.5">
              <FileCode2 className="h-4 w-4" />
              <span>Benchmarks .JSON</span>
            </div>
            <p className="mt-0.5 text-[11px] opacity-75">Timing, RAM & PSNR log</p>
          </div>
          <Download className="h-4 w-4" />
        </a>
      </div>
    </div>
  );
}

