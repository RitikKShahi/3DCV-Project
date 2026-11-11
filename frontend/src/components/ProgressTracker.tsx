"use client";

import type { JobResponse } from "@/lib/types";
import { Activity, Camera, Cpu, Layers, Timer } from "lucide-react";

export default function ProgressTracker({ job }: { job: JobResponse }) {
  const pct = Math.round((job.progress || 0) * 100);
  const badgeColor =
    job.status === "completed"
      ? "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30"
      : job.status === "failed"
      ? "bg-rose-500/15 text-rose-300 ring-rose-500/30"
      : "bg-sky-500/15 text-sky-300 ring-sky-500/30";

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase ring-1 ${badgeColor}`}>
            {job.status}
          </span>
          <span className="text-sm font-medium text-slate-200">{job.stage}</span>
        </div>
        <span className="font-mono text-sm font-semibold text-sky-400">{pct}%</span>
      </div>

      <div className="mt-3 h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full transition-all duration-300 ${
            job.status === "failed" ? "bg-rose-500" : job.status === "completed" ? "bg-emerald-500" : "bg-sky-500"
          }`}
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>

      <p className="mt-2.5 text-xs text-slate-300">{job.message}</p>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
        <div className="rounded-lg bg-slate-950/70 p-3 ring-1 ring-slate-800">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Layers className="h-3.5 w-3.5 text-sky-400" />
            <span>Batch</span>
          </div>
          <p className="mt-1 font-mono text-base font-semibold text-white">
            {job.batch_index} / {job.total_batches}
          </p>
        </div>

        <div className="rounded-lg bg-slate-950/70 p-3 ring-1 ring-slate-800">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Cpu className="h-3.5 w-3.5 text-emerald-400" />
            <span>Gaussians</span>
          </div>
          <p className="mt-1 font-mono text-base font-semibold text-white">
            {job.num_gaussians.toLocaleString()}
          </p>
        </div>

        <div className="rounded-lg bg-slate-950/70 p-3 ring-1 ring-slate-800">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Camera className="h-3.5 w-3.5 text-violet-400" />
            <span>Cameras</span>
          </div>
          <p className="mt-1 font-mono text-base font-semibold text-white">{job.num_cameras}</p>
        </div>

        <div className="rounded-lg bg-slate-950/70 p-3 ring-1 ring-slate-800">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Activity className="h-3.5 w-3.5 text-amber-400" />
            <span>PSNR</span>
          </div>
          <p className="mt-1 font-mono text-base font-semibold text-white">
            {job.psnr > 0 ? `${job.psnr.toFixed(2)} dB` : "—"}
          </p>
        </div>

        <div className="rounded-lg bg-slate-950/70 p-3 ring-1 ring-slate-800">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Timer className="h-3.5 w-3.5 text-rose-400" />
            <span>Elapsed</span>
          </div>
          <p className="mt-1 font-mono text-base font-semibold text-white">
            {job.elapsed_sec.toFixed(1)}s
          </p>
        </div>
      </div>

      {job.error_message && (
        <div className="mt-4 rounded-lg border border-rose-500/40 bg-rose-950/50 p-3 text-xs text-rose-200">
          {job.error_message}
        </div>
      )}
    </div>
  );
}

