"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft, Eye, Loader2, RefreshCw } from "lucide-react";
import ExportPanel from "@/components/ExportPanel";
import LivePreviewPanel from "@/components/LivePreviewPanel";
import ProgressTracker from "@/components/ProgressTracker";
import { useJobProgress } from "@/hooks/useJobProgress";
import {
  getBatchPreviewImageUrl,
  getPreviewSplatUrl,
  getProject,
  startReconstruction,
} from "@/lib/api";
import type { ProjectResponse } from "@/lib/types";

export default function ProjectDashboardPage() {
  const params = useParams<{ id: string }>();
  const projectId = params.id;
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);

  const loadProject = async () => {
    try {
      const p = await getProject(projectId);
      setProject(p);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load project");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProject();
  }, [projectId]);

  const { job, previewRevision } = useJobProgress(project?.latest_job?.id, project?.latest_job);

  const handleResume = async () => {
    if (!project) return;
    setResuming(true);
    try {
      await startReconstruction(project.id, {
        max_resolution: 384,
        frame_interval: 12,
        max_keyframes: 16,
        max_gaussians: 4000,
        iterations_per_batch: 160,
        final_iterations: 120,
        initial_batch_size: 6,
        incremental_batch_size: 4,
        cpu_threads: 4,
        device: "auto",
        sfm_backend: "auto",
        sh_degree: 0,
        camera_fov_deg: 0,
        adaptive_early_stop: true,
        resume: true,
      });
      await loadProject();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Resume failed");
    } finally {
      setResuming(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center gap-2 text-slate-400">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span>Loading reconstruction dashboard...</span>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-12 text-center">
        <p className="text-rose-300">{error || "Project not found"}</p>
        <Link href="/projects" className="mt-4 inline-block text-sm text-sky-400 hover:underline">
          Back to Projects
        </Link>
      </div>
    );
  }

  const previewUrl =
    job && (job.has_preview || job.preview_batches.length > 0 || job.status === "completed")
      ? getPreviewSplatUrl(job.id)
      : null;

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <Link href="/projects" className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-white">
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>All Projects</span>
          </Link>
          <h1 className="mt-1 text-2xl font-bold text-white">{project.name}</h1>
          <p className="text-xs text-slate-400">
            Source: {project.video_filename} · {project.video_width}×{project.video_height}
            {project.video_duration_sec > 0 ? ` · ${project.video_duration_sec.toFixed(1)}s` : ""}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {job?.status === "failed" && (
            <button
              type="button"
              onClick={handleResume}
              disabled={resuming}
              className="inline-flex items-center gap-1.5 rounded-lg bg-amber-600 px-4 py-2 text-sm font-semibold text-white hover:bg-amber-500 disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${resuming ? "animate-spin" : ""}`} />
              <span>Resume from Checkpoint</span>
            </button>
          )}
          {previewUrl && (
            <Link
              href={`/viewer/${project.id}`}
              className="inline-flex items-center gap-1.5 rounded-lg bg-sky-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-400"
            >
              <Eye className="h-4 w-4" />
              <span>Open Full 3D Viewer</span>
            </Link>
          )}
        </div>
      </div>

      {job && <ProgressTracker job={job} />}

      {/* Live progressive WebGL 3D preview */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Live Incremental 3D Gaussian Preview
          </h2>
          {job && job.preview_batches.length > 0 && (
            <span className="text-xs text-slate-400">
              Showing {job.status === "completed" ? "final scene" : `preview batch ${job.preview_batches[job.preview_batches.length - 1] + 1}`}
            </span>
          )}
        </div>
        <LivePreviewPanel
          job={job}
          sceneUrl={previewUrl}
          gaussianFormat="splat"
          revision={previewRevision}
          heightClass="h-[480px]"
          label={
            job?.status === "completed"
              ? `Final 3DGS (${job.num_gaussians.toLocaleString()} splats)`
              : job
              ? `Incremental Preview · Batch ${job.batch_index}/${job.total_batches}`
              : undefined
          }
        />
      </div>

      {/* Incremental batch timeline & rendered previews */}
      {job && job.metrics_history.length > 0 && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Incremental Batch Progression
          </h3>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                <tr>
                  <th className="py-2 pr-4">Batch</th>
                  <th className="py-2 pr-4">Cameras</th>
                  <th className="py-2 pr-4">Gaussians</th>
                  <th className="py-2 pr-4">Added</th>
                  <th className="py-2 pr-4">SfM</th>
                  <th className="py-2 pr-4">Loss</th>
                  <th className="py-2 pr-4">PSNR</th>
                  <th className="py-2 pr-4">Time</th>
                  <th className="py-2">Render Preview</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {job.metrics_history.map((m) => (
                  <tr key={m.batch_index}>
                    <td className="py-2.5 pr-4 font-mono font-semibold text-white">#{m.batch_index + 1}</td>
                    <td className="py-2.5 pr-4 font-mono">{m.num_cameras} (+{m.new_cameras})</td>
                    <td className="py-2.5 pr-4 font-mono">{m.num_gaussians.toLocaleString()}</td>
                    <td className="py-2.5 pr-4 font-mono text-emerald-400">+{m.gaussians_added}</td>
                    <td className="py-2.5 pr-4">{m.sfm_backend}</td>
                    <td className="py-2.5 pr-4 font-mono">{m.final_loss.toFixed(4)}</td>
                    <td className="py-2.5 pr-4 font-mono text-sky-300">{m.mean_psnr.toFixed(2)} dB</td>
                    <td className="py-2.5 pr-4 font-mono">{m.elapsed_sec.toFixed(1)}s</td>
                    <td className="py-2.5">
                      {m.preview_image ? (
                        <img
                          src={getBatchPreviewImageUrl(job.id, m.batch_index)}
                          alt={`Batch ${m.batch_index + 1} render`}
                          className="h-12 w-16 rounded border border-slate-700 object-cover"
                        />
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <ExportPanel job={job} />
    </div>
  );
}

