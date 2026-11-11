"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft, Loader2 } from "lucide-react";
import ExportPanel from "@/components/ExportPanel";
import LivePreviewPanel from "@/components/LivePreviewPanel";
import { useJobProgress } from "@/hooks/useJobProgress";
import { getExportUrl, getPreviewSplatUrl, getProject } from "@/lib/api";
import type { ProjectResponse } from "@/lib/types";

export default function SceneViewerPage() {
  const params = useParams<{ id: string }>();
  const projectId = params.id;
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedSource, setSelectedSource] = useState<"latest" | "ply" | number>("latest");

  useEffect(() => {
    getProject(projectId)
      .then(setProject)
      .finally(() => setLoading(false));
  }, [projectId]);

  const { job, previewRevision } = useJobProgress(project?.latest_job?.id, project?.latest_job);

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center gap-2 text-slate-400">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span>Loading 3D scene viewer...</span>
      </div>
    );
  }

  if (!project || !job) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-12 text-center text-slate-300">
        <p>Scene not found.</p>
        <Link href="/projects" className="mt-4 inline-block text-sm text-sky-400 hover:underline">
          Back to Projects
        </Link>
      </div>
    );
  }

  let sceneUrl: string | null = null;
  let format: "splat" | "ply" = "splat";
  let label = "Latest Incremental Preview";

  if (selectedSource === "ply" && job.has_ply) {
    sceneUrl = getExportUrl(job.id, "ply");
    format = "ply";
    label = `Final .PLY (${job.num_gaussians.toLocaleString()} splats)`;
  } else if (typeof selectedSource === "number") {
    sceneUrl = getPreviewSplatUrl(job.id, selectedSource);
    format = "splat";
    label = `Incremental Batch #${selectedSource + 1}`;
  } else if (job.has_splat) {
    sceneUrl = getExportUrl(job.id, "splat");
    format = "splat";
    label = `Final .SPLAT (${job.num_gaussians.toLocaleString()} splats)`;
  } else if (job.has_preview) {
    sceneUrl = getPreviewSplatUrl(job.id);
    format = "splat";
    label = `Live Preview · Batch ${job.batch_index}/${job.total_batches}`;
  }

  return (
    <div className="mx-auto max-w-7xl space-y-5 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <Link
            href={`/projects/${project.id}`}
            className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-white"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>Back to Reconstruction Dashboard</span>
          </Link>
          <h1 className="mt-1 text-xl font-bold text-white">{project.name} — Interactive 3D Viewer</h1>
        </div>

        {/* Progressive batch scrubber */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400">Scene Revision:</span>
          {job.preview_batches.map((bIdx) => (
            <button
              key={bIdx}
              type="button"
              onClick={() => setSelectedSource(bIdx)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium ${
                selectedSource === bIdx
                  ? "bg-sky-500 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              Batch {bIdx + 1}
            </button>
          ))}
          <button
            type="button"
            onClick={() => setSelectedSource("latest")}
            className={`rounded-md px-2.5 py-1 text-xs font-medium ${
              selectedSource === "latest"
                ? "bg-emerald-600 text-white"
                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
            }`}
          >
            {job.status === "completed" ? "Final .SPLAT" : "Latest"}
          </button>
          {job.has_ply && (
            <button
              type="button"
              onClick={() => setSelectedSource("ply")}
              className={`rounded-md px-2.5 py-1 text-xs font-medium ${
                selectedSource === "ply"
                  ? "bg-violet-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              Final .PLY
            </button>
          )}
        </div>
      </div>

      <LivePreviewPanel
        job={job}
        sceneUrl={sceneUrl}
        gaussianFormat={format}
        revision={previewRevision}
        heightClass="h-[640px]"
        label={label}
      />

      <ExportPanel job={job} />
    </div>
  );
}

