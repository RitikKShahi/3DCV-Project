"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Box, Eye, Plus, Trash2, Loader2 } from "lucide-react";
import { deleteProject, listProjects } from "@/lib/api";
import type { ProjectResponse } from "@/lib/types";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      setLoading(true);
      const data = await listProjects();
      setProjects(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load projects");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleDelete = async (id: string) => {
    await deleteProject(id);
    setProjects((prev) => prev.filter((p) => p.id !== id));
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Reconstruction Projects</h1>
          <p className="mt-1 text-sm text-slate-400">
            Monitor active CPU reconstruction jobs, explore 3D Gaussian scenes, and export outputs.
          </p>
        </div>
        <Link
          href="/upload"
          className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-4 py-2.5 text-sm font-semibold text-white hover:bg-sky-400"
        >
          <Plus className="h-4 w-4" />
          <span>New Reconstruction</span>
        </Link>
      </div>

      {error && (
        <div className="mt-6 rounded-lg border border-rose-500/40 bg-rose-950/60 p-4 text-sm text-rose-200">
          {error}
        </div>
      )}

      {loading ? (
        <div className="mt-12 flex items-center justify-center gap-2 text-slate-400">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span>Loading projects...</span>
        </div>
      ) : projects.length === 0 ? (
        <div className="mt-12 rounded-xl border border-dashed border-slate-800 bg-slate-900/40 p-12 text-center">
          <Box className="mx-auto h-10 w-10 text-slate-500" />
          <p className="mt-3 text-base font-medium text-slate-200">No reconstruction projects yet</p>
          <p className="mt-1 text-sm text-slate-400">Upload a real-world video to start your first CPU 3DGS scan.</p>
          <Link
            href="/upload"
            className="mt-5 inline-flex items-center gap-2 rounded-lg bg-sky-500 px-5 py-2.5 text-sm font-semibold text-white hover:bg-sky-400"
          >
            Upload Video
          </Link>
        </div>
      ) : (
        <div className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((p) => {
            const job = p.latest_job;
            return (
              <div
                key={p.id}
                className="flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-900/60 p-5"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <h2 className="font-semibold text-white">{p.name}</h2>
                    <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-medium text-sky-300">
                      {job?.status || p.status}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-400">
                    {p.video_filename} ({p.video_width}×{p.video_height}
                    {p.video_duration_sec > 0 ? `, ${p.video_duration_sec.toFixed(1)}s` : ""})
                  </p>

                  {job && (
                    <div className="mt-4 space-y-1.5 text-xs text-slate-300">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Progress:</span>
                        <span>{Math.round(job.progress * 100)}% ({job.stage})</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Gaussians:</span>
                        <span className="font-mono">{job.num_gaussians.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">PSNR:</span>
                        <span className="font-mono">{job.psnr > 0 ? `${job.psnr.toFixed(2)} dB` : "—"}</span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="mt-5 flex items-center justify-between gap-2 border-t border-slate-800 pt-4">
                  <div className="flex items-center gap-2">
                    <Link
                      href={`/projects/${p.id}`}
                      className="rounded-md bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
                    >
                      Dashboard
                    </Link>
                    {job?.has_preview && (
                      <Link
                        href={`/viewer/${p.id}`}
                        className="inline-flex items-center gap-1 rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
                      >
                        <Eye className="h-3.5 w-3.5" />
                        <span>3D View</span>
                      </Link>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => handleDelete(p.id)}
                    title="Delete project"
                    className="rounded-md p-1.5 text-slate-400 hover:bg-rose-950/60 hover:text-rose-300"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

