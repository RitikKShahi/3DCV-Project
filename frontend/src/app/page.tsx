import Link from "next/link";
import { ArrowRight, Cpu, Layers, Sparkles, Video, Box } from "lucide-react";

export default function LandingPage() {
  const steps = [
    {
      icon: Video,
      title: "1. Smart Video Preprocessing",
      desc: "FFmpeg + OpenCV extract sharp, non-redundant keyframes with configurable resolution capping to protect CPU RAM.",
    },
    {
      icon: Layers,
      title: "2. Incremental SfM + Consistent Frame",
      desc: "CPU pycolmap (with OpenCV+SciPy fallback) estimates intrinsics, poses, and sparse 3D points in one unified coordinate frame.",
    },
    {
      icon: Cpu,
      title: "3. Pure PyTorch CPU 3DGS Engine",
      desc: "Tile-based sparse differentiable EWA splatting rasterizer optimizes positions, covariances, opacities, and colors on CPU.",
    },
    {
      icon: Box,
      title: "4. Progressive WebGL Exploration & Export",
      desc: "Inspect intermediate reconstructions batch-by-batch in Three.js and export standard .PLY or compact .SPLAT scenes.",
    },
  ];

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:py-20">
      <div className="mx-auto max-w-3xl text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-500/10 px-3.5 py-1 text-xs font-medium text-sky-300">
          <Sparkles className="h-3.5 w-3.5" />
          <span>100% CPU-Only Reconstruction · Zero CUDA Required</span>
        </div>

        <h1 className="mt-6 text-4xl font-bold tracking-tight text-white sm:text-5xl">
          Photorealistic 3D Gaussian Splatting from Real-World Video
        </h1>
        <p className="mt-5 text-base leading-relaxed text-slate-300 sm:text-lg">
          Upload an MP4 or MOV capture and watch your 3D scene assemble incrementally across frame batches—powered
          by a CPU-native SfM + vectorized PyTorch differentiable splatting pipeline and interactive WebGL viewer.
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/upload"
            className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-sky-500/20 hover:bg-sky-400"
          >
            <span>Upload Video & Reconstruct</span>
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            href="/projects"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-6 py-3 text-sm font-semibold text-slate-200 hover:bg-slate-800"
          >
            <span>Open Projects Dashboard</span>
          </Link>
        </div>
      </div>

      <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {steps.map((s) => (
          <div
            key={s.title}
            className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 transition hover:border-slate-700"
          >
            <div className="inline-flex rounded-lg bg-sky-500/10 p-2.5 text-sky-400 ring-1 ring-sky-500/20">
              <s.icon className="h-5 w-5" />
            </div>
            <h2 className="mt-4 text-base font-semibold text-white">{s.title}</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">{s.desc}</p>
          </div>
        ))}
      </div>

      <div className="mt-12 rounded-xl border border-amber-500/20 bg-amber-500/5 p-6 text-sm text-amber-200/90">
        <strong className="font-semibold text-amber-200">CPU Performance Note:</strong> Because all differentiable
        Gaussian splatting runs strictly on CPU without NVIDIA CUDA kernels, default settings cap resolution at
        384px and 4,000 Gaussians so reconstructions complete in minutes within 8 GB RAM. You can tune every
        parameter on the upload page.
      </div>
    </div>
  );
}

