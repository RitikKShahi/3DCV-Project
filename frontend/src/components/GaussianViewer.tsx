"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Maximize2, Minimize2, RotateCcw, Loader2 } from "lucide-react";

interface GaussianViewerProps {
  sceneUrl: string | null;
  format?: "splat" | "ply";
  revision?: number;
  className?: string;
  heightClass?: string;
  label?: string;
  fov?: number;
  onFovChange?: (fov: number) => void;
}

function computeRobustSceneFrame(
  viewer: import("@mkkellogg/gaussian-splats-3d").Viewer
): { pos: [number, number, number]; target: [number, number, number] } | null {
  const mesh = viewer.splatMesh;
  if (!mesh || typeof mesh.getSplatCount !== "function") return null;
  const total = mesh.getSplatCount();
  if (total <= 0) return null;

  const sampleTarget = Math.min(total, 1500);
  const step = Math.max(1, Math.floor(total / sampleTarget));
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const tmp = new THREE.Vector3();

  for (let i = 0; i < total; i += step) {
    mesh.getSplatCenter(i, tmp);
    if (Number.isFinite(tmp.x) && Number.isFinite(tmp.y) && Number.isFinite(tmp.z)) {
      xs.push(tmp.x);
      ys.push(tmp.y);
      zs.push(tmp.z);
    }
  }
  if (xs.length === 0) return null;

  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);

  const q = (arr: number[], p: number) => arr[Math.min(arr.length - 1, Math.max(0, Math.floor(arr.length * p)))];
  const cx = q(xs, 0.5);
  const cy = q(ys, 0.5);
  const cz = q(zs, 0.5);

  const spanX = q(xs, 0.9) - q(xs, 0.1);
  const spanY = q(ys, 0.9) - q(ys, 0.1);
  const spanZ = q(zs, 0.9) - q(zs, 0.1);
  const radius = Math.max(1.2, Math.hypot(spanX, spanY, spanZ) * 0.5);

  return {
    target: [cx, cy, cz],
    pos: [cx, cy - radius * 0.18, cz - Math.max(2.4, radius * 1.15)],
  };
}

export default function GaussianViewer({
  sceneUrl,
  format = "splat",
  revision = 0,
  className = "",
  heightClass = "h-[520px]",
  label,
  fov: controlledFov,
  onFovChange,
}: GaussianViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mountRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<import("@mkkellogg/gaussian-splats-3d").Viewer | null>(null);
  const hasFramedRef = useRef(false);
  const defaultFrameRef = useRef<{ pos: [number, number, number]; target: [number, number, number] }>({
    pos: [0, -0.5, -3.2],
    target: [0, 0, 0],
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [internalFov, setInternalFov] = useState(65);
  const fov = controlledFov ?? internalFov;
  const setFov = (val: number) => {
    const clamped = Math.max(25, Math.min(120, val));
    if (onFovChange) onFovChange(clamped);
    else setInternalFov(clamped);
  };

  // Track WASD / Arrow keys for smooth free camera movement
  useEffect(() => {
    const pressed = new Set<string>();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      pressed.add(e.key.toLowerCase());
    };
    const onKeyUp = (e: KeyboardEvent) => {
      pressed.delete(e.key.toLowerCase());
    };

    let rafId = 0;
    const stepMove = () => {
      const viewer = viewerRef.current;
      if (viewer && pressed.size > 0 && viewer.camera && viewer.controls) {
        const speed = 0.08;
        const forward = new THREE.Vector3();
        viewer.camera.getWorldDirection(forward);
        const right = new THREE.Vector3().crossVectors(forward, viewer.camera.up).normalize();
        const up = viewer.camera.up.clone().normalize();
        const delta = new THREE.Vector3();

        if (pressed.has("w") || pressed.has("arrowup")) delta.addScaledVector(forward, speed);
        if (pressed.has("s") || pressed.has("arrowdown")) delta.addScaledVector(forward, -speed);
        if (pressed.has("d") || pressed.has("arrowright")) delta.addScaledVector(right, speed);
        if (pressed.has("a") || pressed.has("arrowleft")) delta.addScaledVector(right, -speed);
        if (pressed.has("e") || pressed.has(" ")) delta.addScaledVector(up, speed);
        if (pressed.has("q") || pressed.has("shift")) delta.addScaledVector(up, -speed);

        if (delta.lengthSq() > 0) {
          viewer.camera.position.add(delta);
          viewer.controls.target.add(delta);
          viewer.controls.update();
        }
      }
      rafId = requestAnimationFrame(stepMove);
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    rafId = requestAnimationFrame(stepMove);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  // Apply live FOV slider updates to the active Three.js camera
  useEffect(() => {
    const viewer = viewerRef.current;
    if (viewer?.camera) {
      viewer.camera.fov = fov;
      viewer.camera.updateProjectionMatrix();
    }
  }, [fov]);

  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  useEffect(() => {
    if (!sceneUrl || !mountRef.current) return;

    let cancelled = false;
    const mountEl = mountRef.current;

    const initViewer = async () => {
      setLoading(true);
      setError(null);

      // Preserve user's camera pose across progressive preview reloads once framed
      let prevPos: [number, number, number] = defaultFrameRef.current.pos;
      let prevTarget: [number, number, number] = defaultFrameRef.current.target;
      if (hasFramedRef.current && viewerRef.current?.camera && viewerRef.current?.controls) {
        const p = viewerRef.current.camera.position;
        const t = viewerRef.current.controls.target;
        prevPos = [p.x, p.y, p.z];
        prevTarget = [t.x, t.y, t.z];
      }

      if (viewerRef.current) {
        try {
          await viewerRef.current.dispose();
        } catch {
          // ignore dispose error
        }
        viewerRef.current = null;
      }
      mountEl.innerHTML = "";

      try {
        const GS3D = await import("@mkkellogg/gaussian-splats-3d");
        if (cancelled) return;

        const viewer = new GS3D.Viewer({
          rootElement: mountEl,
          cameraUp: [0, -1, 0], // COLMAP / OpenCV Y-down coordinate convention
          initialCameraPosition: prevPos,
          initialCameraLookAt: prevTarget,
          sharedMemoryForWorkers: false,
          useBuiltInControls: true,
          sceneRevealMode: GS3D.SceneRevealMode.Instant,
          logLevel: GS3D.LogLevel.None,
        });
        viewerRef.current = viewer;

        const cacheBustedUrl = `${sceneUrl}${sceneUrl.includes("?") ? "&" : "?"}rev=${revision}`;
        await viewer.addSplatScene(cacheBustedUrl, {
          format: format === "ply" ? GS3D.SceneFormat.Ply : GS3D.SceneFormat.Splat,
          splatAlphaRemovalThreshold: 2,
          showLoadingUI: false,
        });

        if (!cancelled) {
          if (viewer.camera) {
            viewer.camera.fov = fov;
            viewer.camera.updateProjectionMatrix();
          }

          const autoFrame = computeRobustSceneFrame(viewer);
          if (autoFrame) {
            defaultFrameRef.current = autoFrame;
            if (!hasFramedRef.current && viewer.camera && viewer.controls) {
              viewer.camera.position.set(...autoFrame.pos);
              viewer.controls.target.set(...autoFrame.target);
              viewer.controls.update();
              hasFramedRef.current = true;
            }
          }

          viewer.start();
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load 3D Gaussian scene");
          setLoading(false);
        }
      }
    };

    initViewer();

    return () => {
      cancelled = true;
      if (viewerRef.current) {
        viewerRef.current.dispose().catch(() => {});
        viewerRef.current = null;
      }
    };
  }, [sceneUrl, format, revision]);

  const handleResetCamera = () => {
    const viewer = viewerRef.current;
    if (!viewer || !viewer.camera || !viewer.controls) return;
    const autoFrame = computeRobustSceneFrame(viewer) || defaultFrameRef.current;
    defaultFrameRef.current = autoFrame;
    viewer.camera.position.set(...autoFrame.pos);
    viewer.controls.target.set(...autoFrame.target);
    viewer.controls.update();
  };

  const toggleFullscreen = async () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      await containerRef.current.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  };

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950 ${heightClass} ${className}`}
    >
      <div ref={mountRef} className="h-full w-full" />

      {!sceneUrl && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-slate-400">
          <p className="text-sm font-medium">Waiting for first incremental Gaussian preview...</p>
          <p className="mt-1 text-xs text-slate-500">
            The 3D viewer will automatically populate once Batch 1 finishes optimizing.
          </p>
        </div>
      )}

      {loading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-slate-950/60 backdrop-blur-sm">
          <div className="flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm text-sky-300 ring-1 ring-slate-700">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Loading Gaussian Splats...</span>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 p-6 text-center">
          <p className="max-w-md rounded-lg bg-rose-950/70 px-4 py-3 text-sm text-rose-200 ring-1 ring-rose-700/60">
            Viewer error: {error}
          </p>
        </div>
      )}

      {/* Overlay controls */}
      <div className="pointer-events-none absolute inset-x-3 top-3 flex flex-wrap items-center justify-between gap-2">
        {label ? (
          <span className="pointer-events-auto rounded-md bg-slate-900/80 px-2.5 py-1 text-xs font-medium text-slate-200 ring-1 ring-slate-700/80 backdrop-blur">
            {label}
          </span>
        ) : (
          <span />
        )}
        <div className="pointer-events-auto flex items-center gap-1.5">
          <label className="flex items-center gap-1.5 rounded-md bg-slate-900/80 px-2 py-1 text-[11px] text-slate-200 ring-1 ring-slate-700/80 backdrop-blur">
            <span>FOV {Math.round(fov)}°</span>
            <input
              type="range"
              min={25}
              max={120}
              step={1}
              value={fov}
              onChange={(e) => setFov(Number(e.target.value))}
              className="w-16"
            />
          </label>
          <button
            type="button"
            onClick={handleResetCamera}
            title="Fit Scene / Reset Camera"
            className="rounded-md bg-slate-900/80 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 ring-1 ring-slate-700/80 backdrop-blur hover:bg-slate-800 inline-flex items-center gap-1"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Fit</span>
          </button>
          <button
            type="button"
            onClick={toggleFullscreen}
            title="Toggle Fullscreen"
            className="rounded-md bg-slate-900/80 p-2 text-slate-200 ring-1 ring-slate-700/80 backdrop-blur hover:bg-slate-800"
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
        </div>
      </div>

      <div className="pointer-events-none absolute bottom-3 left-3 hidden rounded-md bg-slate-900/75 px-2.5 py-1 text-[11px] text-slate-300 ring-1 ring-slate-800 backdrop-blur sm:block">
        Drag: Orbit · Right-Drag: Pan · Scroll: Zoom · WASD / Q / E: Free Move
      </div>
    </div>
  );
}
