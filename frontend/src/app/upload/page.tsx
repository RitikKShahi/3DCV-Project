"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { UploadCloud, Sliders, Play, Loader2, Video, Images, Cpu } from "lucide-react";
import FovPreviewWidget from "@/components/FovPreviewWidget";
import { getHealth, startReconstruction, uploadPhotos, uploadVideo } from "@/lib/api";
import type { HealthResponse, ReconstructionParams } from "@/lib/types";

const DEFAULT_PARAMS: ReconstructionParams = {
  max_resolution: 384,
  frame_interval: 10,
  max_keyframes: 20,
  max_gaussians: 5000,
  iterations_per_batch: 200,
  final_iterations: 150,
  initial_batch_size: 8,
  incremental_batch_size: 4,
  cpu_threads: 4,
  device: "auto",
  sfm_backend: "auto",
  sh_degree: 0,
  camera_fov_deg: 0,
  adaptive_early_stop: true,
  resume: true,
};

export default function UploadPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"video" | "photos">("video");
  const [file, setFile] = useState<File | null>(null);
  const [photoFiles, setPhotoFiles] = useState<File[]>([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [params, setParams] = useState<ReconstructionParams>(DEFAULT_PARAMS);
  const [submitting, setSubmitting] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [previewThumb, setPreviewThumb] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let active = true;
    getHealth()
      .then((res) => {
        if (active) setHealth(res);
      })
      .catch(() => {
        // ignore health check failure
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (mode === "photos" && photoFiles.length > 0) {
      const url = URL.createObjectURL(photoFiles[0]);
      setPreviewThumb(url);
      return () => URL.revokeObjectURL(url);
    }
    setPreviewThumb(null);
  }, [mode, photoFiles]);

  const updateNum = (key: keyof ReconstructionParams, val: number) => {
    setParams((prev) => ({ ...prev, [key]: val }));
  };

  const totalPhotoBytes = photoFiles.reduce((acc, f) => acc + f.size, 0);
  const hasInput = mode === "video" ? Boolean(file) : photoFiles.length >= 3;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === "video" && !file) {
      setError("Please select a video file first.");
      return;
    }
    if (mode === "photos" && photoFiles.length < 3) {
      setError("Please select at least 3 overlapping photos.");
      return;
    }

    setError(null);
    setSubmitting(true);

    try {
      const project =
        mode === "video" && file
          ? (setStatusMsg("Uploading video to server..."), await uploadVideo(file, name, description))
          : (setStatusMsg(`Uploading ${photoFiles.length} photos to server...`),
            await uploadPhotos(photoFiles, name, description));

      setStatusMsg("Starting incremental CPU reconstruction job...");
      await startReconstruction(project.id, params);
      router.push(`/projects/${project.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload or job creation failed");
      setSubmitting(false);
      setStatusMsg("");
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6">
      <h1 className="text-2xl font-bold text-white">Upload Video or Photos & Configure CPU Reconstruction</h1>
      <p className="mt-1 text-sm text-slate-400">
        Capture tips: smooth orbital views around an object or room with textured surfaces and steady lighting give
        the strongest SfM poses and 3DGS quality.
      </p>

      {/* Mode toggle */}
      <div className="mt-6 inline-flex rounded-xl border border-slate-800 bg-slate-900/70 p-1">
        <button
          type="button"
          onClick={() => {
            setMode("video");
            setError(null);
          }}
          className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
            mode === "video" ? "bg-sky-500 text-white" : "text-slate-300 hover:text-white"
          }`}
        >
          <Video className="h-4 w-4" />
          <span>Video Upload</span>
        </button>
        <button
          type="button"
          onClick={() => {
            setMode("photos");
            setError(null);
          }}
          className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
            mode === "photos" ? "bg-sky-500 text-white" : "text-slate-300 hover:text-white"
          }`}
        >
          <Images className="h-4 w-4" />
          <span>Photo Set Upload</span>
        </button>
      </div>

      <form onSubmit={handleSubmit} className="mt-6 space-y-6">
        {/* Dropzone */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
          {mode === "video" ? (
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-950/60 p-8 text-center transition hover:border-sky-500/60">
              <UploadCloud className="h-10 w-10 text-sky-400" />
              <span className="mt-3 text-sm font-medium text-white">
                {file ? file.name : "Click to select MP4, MOV, AVI, WebM, or MKV video"}
              </span>
              <span className="mt-1 text-xs text-slate-400">
                {file
                  ? `${(file.size / (1024 * 1024)).toFixed(2)} MB ready`
                  : "Recommended: 5–30 seconds handheld or turntable video"}
              </span>
              <input
                type="file"
                accept=".mp4,.mov,.avi,.webm,.mkv,.m4v,video/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0] || null;
                  setFile(f);
                  if (f && !name) setName(f.name.replace(/\.[^.]+$/, ""));
                }}
              />
            </label>
          ) : (
            <div>
              <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-950/60 p-8 text-center transition hover:border-sky-500/60">
                <Images className="h-10 w-10 text-sky-400" />
                <span className="mt-3 text-sm font-medium text-white">
                  {photoFiles.length > 0
                    ? `${photoFiles.length} photos selected`
                    : "Click to select multiple photos (JPG, PNG, WebP, BMP, TIFF)"}
                </span>
                <span className="mt-1 text-xs text-slate-400">
                  {photoFiles.length > 0
                    ? `${(totalPhotoBytes / (1024 * 1024)).toFixed(2)} MB total · select 3–500 overlapping images`
                    : "Select all photos together (Ctrl/Cmd+A or Shift+Click). Minimum 3 photos, up to 500."}
                </span>
                <input
                  type="file"
                  multiple
                  accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,image/*"
                  className="hidden"
                  onChange={(e) => {
                    const selected = Array.from(e.target.files || []);
                    setPhotoFiles(selected);
                    if (selected.length > 0) {
                      if (!name) setName(`Photo Scan (${selected.length} imgs)`);
                      setParams((p) => ({
                        ...p,
                        max_keyframes: Math.min(500, Math.max(4, selected.length)),
                        initial_batch_size: Math.min(p.initial_batch_size, Math.max(3, Math.ceil(selected.length / 2))),
                      }));
                    }
                  }}
                />
              </label>

              {photoFiles.length > 0 && (
                <div className="mt-3 max-h-28 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950/70 px-3 py-2 text-xs text-slate-300">
                  <div className="flex flex-wrap gap-1.5">
                    {photoFiles.slice(0, 24).map((pf, idx) => (
                      <span key={`${pf.name}-${idx}`} className="rounded bg-slate-800 px-2 py-0.5 text-[11px]">
                        {pf.name}
                      </span>
                    ))}
                    {photoFiles.length > 24 && (
                      <span className="px-1 text-slate-400">+{photoFiles.length - 24} more</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="mt-5 grid gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-medium text-slate-300">Project Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={mode === "video" ? "e.g., Desk Sculpture Scan" : "e.g., Figurine Multi-View Photos"}
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-300">Description (optional)</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Indoor object orbit"
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* CPU & Reconstruction parameters */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Sliders className="h-4 w-4 text-sky-400" />
            <span>CPU Performance & Pipeline Parameters</span>
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            <div>
              <label className="block text-xs text-slate-300">Max Resolution (px)</label>
              <input
                type="number"
                min={160}
                max={1280}
                step={32}
                value={params.max_resolution}
                onChange={(e) => updateNum("max_resolution", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300">
                Frame Extraction Interval {mode === "photos" && "(Video only)"}
              </label>
              <input
                type="number"
                min={1}
                max={120}
                disabled={mode === "photos"}
                value={params.frame_interval}
                onChange={(e) => updateNum("frame_interval", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white disabled:opacity-40"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300">
                {mode === "photos" ? "Max Photos to Use (up to 500)" : "Max Keyframes (up to 500)"}
              </label>
              <input
                type="number"
                min={4}
                max={500}
                value={params.max_keyframes}
                onChange={(e) => updateNum("max_keyframes", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300">Max Gaussian Count</label>
              <input
                type="number"
                min={500}
                max={50000}
                step={500}
                value={params.max_gaussians}
                onChange={(e) => updateNum("max_gaussians", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300">Iterations / Batch</label>
              <input
                type="number"
                min={20}
                max={3000}
                step={20}
                value={params.iterations_per_batch}
                onChange={(e) => updateNum("iterations_per_batch", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300">Final Refinement Iters</label>
              <input
                type="number"
                min={0}
                max={2000}
                step={20}
                value={params.final_iterations}
                onChange={(e) => updateNum("final_iterations", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300">Initial Batch Size</label>
              <input
                type="number"
                min={3}
                max={100}
                value={params.initial_batch_size}
                onChange={(e) => updateNum("initial_batch_size", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300">Incremental Batch Size</label>
              <input
                type="number"
                min={1}
                max={50}
                value={params.incremental_batch_size}
                onChange={(e) => updateNum("incremental_batch_size", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300">CPU Threads</label>
              <input
                type="number"
                min={1}
                max={16}
                value={params.cpu_threads}
                onChange={(e) => updateNum("cpu_threads", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="flex items-center justify-between text-xs text-slate-300">
                <span className="inline-flex items-center gap-1">
                  <Cpu className="h-3.5 w-3.5 text-sky-400" />
                  Compute Device
                </span>
                {health && (
                  <span className="text-[11px] text-emerald-400">
                    {health.cuda_available
                      ? "CUDA GPU detected"
                      : health.mps_available
                        ? "Apple MPS GPU detected"
                        : `${health.cpu_cores}-core CPU`}
                  </span>
                )}
              </label>
              <select
                value={params.device}
                onChange={(e) =>
                  setParams((p) => ({ ...p, device: e.target.value as ReconstructionParams["device"] }))
                }
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              >
                <option value="auto">
                  Auto (
                  {health?.recommended_device
                    ? `uses ${health.recommended_device.toUpperCase()}`
                    : "GPU if available, else CPU"}
                  )
                </option>
                <option value="mps" disabled={Boolean(health && !health.mps_available)}>
                  GPU — Apple Silicon (MPS)
                  {health && !health.mps_available ? " (unavailable)" : ""}
                </option>
                <option value="cuda" disabled={Boolean(health && !health.cuda_available)}>
                  GPU — NVIDIA (CUDA)
                  {health && !health.cuda_available ? " (unavailable)" : ""}
                </option>
                <option value="cpu">CPU Only</option>
              </select>
              {health?.mps_detected && !health.mps_available && health.mps_note && (
                <p className="mt-1 text-[11px] text-amber-400/90">{health.mps_note}</p>
              )}
            </div>

            <div>
              <label className="block text-xs text-slate-300">SfM Engine</label>
              <select
                value={params.sfm_backend}
                onChange={(e) =>
                  setParams((p) => ({ ...p, sfm_backend: e.target.value as ReconstructionParams["sfm_backend"] }))
                }
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              >
                <option value="auto">Auto (pycolmap + OpenCV fallback)</option>
                <option value="pycolmap">pycolmap SfM</option>
                <option value="opencv">OpenCV + SciPy BA</option>
              </select>
            </div>

            <div>
              <label className="block text-xs text-slate-300">SH Degree</label>
              <select
                value={params.sh_degree}
                onChange={(e) => updateNum("sh_degree", Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
              >
                <option value={0}>Degree 0 (Fast RGB)</option>
                <option value={1}>Degree 1 (View-dependent)</option>
                <option value={3}>Degree 3 (Full SH)</option>
              </select>
            </div>

            <div className="flex items-end pb-2">
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={params.adaptive_early_stop}
                  onChange={(e) => setParams((p) => ({ ...p, adaptive_early_stop: e.target.checked }))}
                  className="rounded border-slate-700 bg-slate-950"
                />
                <span>Adaptive Early Stopping</span>
              </label>
            </div>
          </div>

          <FovPreviewWidget
            fovDeg={params.camera_fov_deg}
            onChange={(val) => updateNum("camera_fov_deg", val)}
            previewImageUrl={previewThumb}
          />
        </div>

        {error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/60 p-3 text-sm text-rose-200">
            {error}
          </div>
        )}

        <div className="flex items-center justify-end gap-3">
          <button
            type="submit"
            disabled={submitting || !hasInput}
            className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-6 py-3 text-sm font-semibold text-white hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            <span>
              {submitting
                ? statusMsg || "Starting..."
                : mode === "video"
                ? "Upload Video & Start Incremental 3DGS"
                : `Upload ${photoFiles.length || ""} Photos & Start Incremental 3DGS`}
            </span>
          </button>
        </div>
      </form>
    </div>
  );
}
