"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Loader2, Maximize2, Minimize2, RotateCcw } from "lucide-react";
import { fetchRepresentation } from "@/lib/api";
import type {
  GeometryRepresentation,
  MeshRepresentation,
  TriangleSplatsRepresentation,
} from "@/lib/types";

interface MultiModeSceneViewerProps {
  jobId: string | null | undefined;
  mode: "geometry" | "mesh" | "triangle_splats";
  revision?: number;
  hasData?: boolean;
  heightClass?: string;
  label?: string;
  fov?: number;
  onFovChange?: (fov: number) => void;
  onStats?: (statsText: string) => void;
}

export default function MultiModeSceneViewer({
  jobId,
  mode,
  revision = 0,
  hasData = false,
  heightClass = "h-[480px]",
  label,
  fov: controlledFov,
  onFovChange,
  onStats,
}: MultiModeSceneViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mountRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const defaultTargetRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const defaultPosRef = useRef<THREE.Vector3>(new THREE.Vector3(0, -0.6, -3.2));

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Mode-specific visual controls
  const [showCameras, setShowCameras] = useState(true);
  const [showAxes, setShowAxes] = useState(false);
  const [pointSize, setPointSize] = useState(0.045);
  const [meshWireframe, setMeshWireframe] = useState<"shaded" | "wireframe" | "both">("shaded");
  const [softTriSplats, setSoftTriSplats] = useState(true);
  const [triScale, setTriScale] = useState(1.0);
  const [internalFov, setInternalFov] = useState(65);
  const fov = controlledFov ?? internalFov;
  const setFov = (val: number) => {
    const clamped = Math.max(25, Math.min(120, val));
    if (onFovChange) onFovChange(clamped);
    else setInternalFov(clamped);
  };

  const [repData, setRepData] = useState<
    GeometryRepresentation | MeshRepresentation | TriangleSplatsRepresentation | null
  >(null);

  // Fetch representation JSON from backend when job/mode/revision updates
  useEffect(() => {
    if (!jobId || !hasData) {
      setRepData(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchRepresentation(jobId, mode)
      .then((data) => {
        if (!cancelled) {
          setRepData(data);
          setLoading(false);
          if (onStats) {
            if (data.mode === "geometry") {
              onStats(`${data.count.toLocaleString()} points · ${data.cameras.length} cameras`);
            } else if (data.mode === "mesh") {
              onStats(`${data.vertex_count.toLocaleString()} verts · ${data.face_count.toLocaleString()} faces`);
            } else {
              onStats(`${data.triangle_count.toLocaleString()} oriented triangle splats`);
            }
          }
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load representation");
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [jobId, mode, revision, hasData]);

  // WASD / Q / E free camera movement
  useEffect(() => {
    const pressed = new Set<string>();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      pressed.add(e.key.toLowerCase());
    };
    const onKeyUp = (e: KeyboardEvent) => pressed.delete(e.key.toLowerCase());

    let raf = 0;
    const tick = () => {
      const cam = cameraRef.current;
      const ctrl = controlsRef.current;
      if (cam && ctrl && pressed.size > 0) {
        const speed = 0.06;
        const forward = new THREE.Vector3();
        cam.getWorldDirection(forward);
        const right = new THREE.Vector3().crossVectors(forward, cam.up).normalize();
        const up = cam.up.clone().normalize();
        const delta = new THREE.Vector3();
        if (pressed.has("w") || pressed.has("arrowup")) delta.addScaledVector(forward, speed);
        if (pressed.has("s") || pressed.has("arrowdown")) delta.addScaledVector(forward, -speed);
        if (pressed.has("d") || pressed.has("arrowright")) delta.addScaledVector(right, speed);
        if (pressed.has("a") || pressed.has("arrowleft")) delta.addScaledVector(right, -speed);
        if (pressed.has("e") || pressed.has(" ")) delta.addScaledVector(up, speed);
        if (pressed.has("q") || pressed.has("shift")) delta.addScaledVector(up, -speed);
        if (delta.lengthSq() > 0) {
          cam.position.add(delta);
          ctrl.target.add(delta);
          ctrl.update();
        }
      }
      raf = requestAnimationFrame(tick);
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  useEffect(() => {
    const onFs = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  // Build Three.js scene whenever repData or mode visual toggles change
  useEffect(() => {
    const mountEl = mountRef.current;
    if (!mountEl || !repData) return;

    const w = mountEl.clientWidth || 640;
    const h = mountEl.clientHeight || 480;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x060911, 1);

    mountEl.innerHTML = "";
    mountEl.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(fov, w / h, 0.02, 500);
    camera.up.set(0, -1, 0); // Match COLMAP Y-down convention

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Preserve prior camera pose across toggle changes if already positioned
    const hadCamera = cameraRef.current !== null && controlsRef.current !== null;
    if (hadCamera && cameraRef.current && controlsRef.current) {
      camera.position.copy(cameraRef.current.position);
      controls.target.copy(controlsRef.current.target);
    }

    cameraRef.current = camera;
    controlsRef.current = controls;

    // Lighting for mesh / shaded modes
    const amb = new THREE.AmbientLight(0xffffff, 0.75);
    const dir1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dir1.position.set(2, -4, -3);
    const dir2 = new THREE.DirectionalLight(0x93c5fd, 0.45);
    dir2.position.set(-3, 2, 3);
    scene.add(amb, dir1, dir2);

    const rootGroup = new THREE.Group();
    scene.add(rootGroup);

    let centerSum = new THREE.Vector3();
    let centerCount = 0;

    if (repData.mode === "geometry") {
      const posArr = new Float32Array(repData.positions);
      const colArr = new Float32Array(repData.colors);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
      geom.setAttribute("color", new THREE.BufferAttribute(colArr, 3));

      const mat = new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: true,
        sizeAttenuation: true,
      });
      rootGroup.add(new THREE.Points(geom, mat));

      for (let i = 0; i < posArr.length; i += 3) {
        centerSum.x += posArr[i];
        centerSum.y += posArr[i + 1];
        centerSum.z += posArr[i + 2];
        centerCount++;
      }

      // Optional Gaussian ellipsoid principal axes wireframe
      if (showAxes && repData.count > 0) {
        const maxShow = Math.min(repData.count, 1500);
        const linePos = new Float32Array(maxShow * 6 * 3);
        const lineCol = new Float32Array(maxShow * 6 * 3);
        const q = new THREE.Quaternion();
        const m = new THREE.Matrix4();
        const ex = new THREE.Vector3();
        const ey = new THREE.Vector3();
        const ez = new THREE.Vector3();

        for (let i = 0; i < maxShow; i++) {
          const px = posArr[i * 3];
          const py = posArr[i * 3 + 1];
          const pz = posArr[i * 3 + 2];
          const sx = repData.scales[i * 3] * 1.5;
          const sy = repData.scales[i * 3 + 1] * 1.5;
          const sz = repData.scales[i * 3 + 2] * 1.5;
          // stored quat is (w, x, y, z)
          q.set(
            repData.rotations[i * 4 + 1],
            repData.rotations[i * 4 + 2],
            repData.rotations[i * 4 + 3],
            repData.rotations[i * 4]
          );
          m.makeRotationFromQuaternion(q);
          ex.set(1, 0, 0).applyMatrix4(m).multiplyScalar(sx);
          ey.set(0, 1, 0).applyMatrix4(m).multiplyScalar(sy);
          ez.set(0, 0, 1).applyMatrix4(m).multiplyScalar(sz);

          const base = i * 18;
          const cr = colArr[i * 3];
          const cg = colArr[i * 3 + 1];
          const cb = colArr[i * 3 + 2];

          const axes = [ex, ey, ez];
          for (let a = 0; a < 3; a++) {
            const off = base + a * 6;
            linePos[off] = px - axes[a].x;
            linePos[off + 1] = py - axes[a].y;
            linePos[off + 2] = pz - axes[a].z;
            linePos[off + 3] = px + axes[a].x;
            linePos[off + 4] = py + axes[a].y;
            linePos[off + 5] = pz + axes[a].z;
            for (let k = 0; k < 2; k++) {
              lineCol[off + k * 3] = cr;
              lineCol[off + k * 3 + 1] = cg;
              lineCol[off + k * 3 + 2] = cb;
            }
          }
        }
        const axesGeom = new THREE.BufferGeometry();
        axesGeom.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
        axesGeom.setAttribute("color", new THREE.BufferAttribute(lineCol, 3));
        rootGroup.add(new THREE.LineSegments(axesGeom, new THREE.LineBasicMaterial({ vertexColors: true, opacity: 0.7, transparent: true })));
      }

      // Optional Camera Frustums
      if (showCameras && repData.cameras.length > 0) {
        const camLines: number[] = [];
        const s = 0.16;
        for (const c of repData.cameras) {
          const O = new THREE.Vector3(...c.center);
          const F = new THREE.Vector3(...c.forward).multiplyScalar(s * 1.4);
          const U = new THREE.Vector3(...c.up).multiplyScalar(s * 0.7);
          const R = new THREE.Vector3(...c.right).multiplyScalar(s);
          const tl = O.clone().add(F).add(U).sub(R);
          const tr = O.clone().add(F).add(U).add(R);
          const br = O.clone().add(F).sub(U).add(R);
          const bl = O.clone().add(F).sub(U).sub(R);
          for (const corner of [tl, tr, br, bl]) {
            camLines.push(O.x, O.y, O.z, corner.x, corner.y, corner.z);
          }
          camLines.push(tl.x, tl.y, tl.z, tr.x, tr.y, tr.z);
          camLines.push(tr.x, tr.y, tr.z, br.x, br.y, br.z);
          camLines.push(br.x, br.y, br.z, bl.x, bl.y, bl.z);
          camLines.push(bl.x, bl.y, bl.z, tl.x, tl.y, tl.z);
        }
        const fg = new THREE.BufferGeometry();
        fg.setAttribute("position", new THREE.Float32BufferAttribute(camLines, 3));
        rootGroup.add(new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ color: 0x38bdf8 })));
      }
    } else if (repData.mode === "mesh") {
      if (repData.vertex_count > 0 && repData.face_count > 0) {
        const vArr = new Float32Array(repData.vertices);
        const nArr = new Float32Array(repData.normals);
        const cArr = new Float32Array(repData.colors);
        const idxArr = new Uint32Array(repData.faces);

        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(vArr, 3));
        geom.setAttribute("normal", new THREE.BufferAttribute(nArr, 3));
        geom.setAttribute("color", new THREE.BufferAttribute(cArr, 3));
        geom.setIndex(new THREE.BufferAttribute(idxArr, 1));

        if (meshWireframe === "shaded" || meshWireframe === "both") {
          const solidMat = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.72,
            metalness: 0.05,
            side: THREE.DoubleSide,
          });
          rootGroup.add(new THREE.Mesh(geom, solidMat));
        }
        if (meshWireframe === "wireframe" || meshWireframe === "both") {
          const wireMat = new THREE.MeshBasicMaterial({
            color: meshWireframe === "wireframe" ? 0x38bdf8 : 0x0f172a,
            wireframe: true,
            transparent: meshWireframe === "both",
            opacity: meshWireframe === "both" ? 0.35 : 1.0,
          });
          rootGroup.add(new THREE.Mesh(geom, wireMat));
        }

        for (let i = 0; i < vArr.length; i += 3) {
          centerSum.x += vArr[i];
          centerSum.y += vArr[i + 1];
          centerSum.z += vArr[i + 2];
          centerCount++;
        }
      }
    } else if (repData.mode === "triangle_splats") {
      if (repData.triangle_count > 0) {
        const rawPos = new Float32Array(repData.positions);
        // Allow live triangle scale adjustment around each triangle's centroid
        const scaledPos = new Float32Array(rawPos.length);
        for (let i = 0; i < rawPos.length; i += 9) {
          const cx = (rawPos[i] + rawPos[i + 3] + rawPos[i + 6]) / 3;
          const cy = (rawPos[i + 1] + rawPos[i + 4] + rawPos[i + 7]) / 3;
          const cz = (rawPos[i + 2] + rawPos[i + 5] + rawPos[i + 8]) / 3;
          centerSum.x += cx;
          centerSum.y += cy;
          centerSum.z += cz;
          centerCount++;
          for (let k = 0; k < 3; k++) {
            const o = i + k * 3;
            scaledPos[o] = cx + (rawPos[o] - cx) * triScale;
            scaledPos[o + 1] = cy + (rawPos[o + 1] - cy) * triScale;
            scaledPos[o + 2] = cz + (rawPos[o + 2] - cz) * triScale;
          }
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(scaledPos, 3));
        geom.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(repData.normals), 3));
        geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(repData.colors), 3));
        geom.setAttribute("aOpacity", new THREE.BufferAttribute(new Float32Array(repData.opacities), 1));
        geom.setAttribute("aLocalUv", new THREE.BufferAttribute(new Float32Array(repData.uvs), 2));

        const mat = new THREE.ShaderMaterial({
          side: THREE.DoubleSide,
          transparent: softTriSplats,
          depthWrite: !softTriSplats,
          uniforms: {
            uSoft: { value: softTriSplats ? 1.0 : 0.0 },
          },
          vertexShader: `
            attribute vec3 color;
            attribute float aOpacity;
            attribute vec2 aLocalUv;
            varying vec3 vColor;
            varying float vOpacity;
            varying vec2 vUv;
            void main() {
              vColor = color;
              vOpacity = aOpacity;
              vUv = aLocalUv;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
          fragmentShader: `
            uniform float uSoft;
            varying vec3 vColor;
            varying float vOpacity;
            varying vec2 vUv;
            void main() {
              if (uSoft > 0.5) {
                float r2 = dot(vUv, vUv);
                float alpha = vOpacity * exp(-2.6 * r2);
                if (alpha < 0.03) discard;
                gl_FragColor = vec4(vColor, alpha);
              } else {
                gl_FragColor = vec4(vColor, 1.0);
              }
            }
          `,
        });

        rootGroup.add(new THREE.Mesh(geom, mat));
      }
    }

    if (centerCount > 0) {
      const target = centerSum.divideScalar(centerCount);
      defaultTargetRef.current.copy(target);
      const box = new THREE.Box3().setFromObject(rootGroup);
      const size = box.getSize(new THREE.Vector3()).length() || 3.0;
      const defaultPos = target.clone().add(new THREE.Vector3(0, -size * 0.15, -Math.max(2.0, size * 0.9)));
      defaultPosRef.current.copy(defaultPos);

      if (!hadCamera) {
        camera.position.copy(defaultPos);
        controls.target.copy(target);
      }
      controls.update();
    }

    const onResize = () => {
      if (!mountEl) return;
      const nw = mountEl.clientWidth || 640;
      const nh = mountEl.clientHeight || 480;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    window.addEventListener("resize", onResize);

    let animId = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
    };
  }, [repData, showCameras, showAxes, pointSize, meshWireframe, softTriSplats, triScale]);

  useEffect(() => {
    if (cameraRef.current) {
      cameraRef.current.fov = fov;
      cameraRef.current.updateProjectionMatrix();
    }
  }, [fov]);

  const handleResetCamera = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.copy(defaultPosRef.current);
    controlsRef.current.target.copy(defaultTargetRef.current);
    controlsRef.current.update();
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
      className={`relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950 ${heightClass}`}
    >
      <div ref={mountRef} className="h-full w-full" />

      {!hasData && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-slate-400">
          <p className="text-sm font-medium">Waiting for first incremental preview...</p>
          <p className="mt-1 text-xs text-slate-500">
            Geometry, Mesh, and Triangle Splatting views populate once Batch 1 finishes optimizing.
          </p>
        </div>
      )}

      {loading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-slate-950/60 backdrop-blur-sm">
          <div className="flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm text-sky-300 ring-1 ring-slate-700">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Building {mode.replace("_", " ")} representation...</span>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 p-6 text-center">
          <p className="max-w-md rounded-lg bg-rose-950/70 px-4 py-3 text-sm text-rose-200 ring-1 ring-rose-700/60">
            {error}
          </p>
        </div>
      )}

      {/* Top overlay controls */}
      <div className="pointer-events-none absolute inset-x-3 top-3 flex flex-wrap items-center justify-between gap-2">
        {label ? (
          <span className="pointer-events-auto rounded-md bg-slate-900/85 px-2.5 py-1 text-xs font-medium text-slate-200 ring-1 ring-slate-700/80 backdrop-blur">
            {label}
          </span>
        ) : (
          <span />
        )}

        <div className="pointer-events-auto flex flex-wrap items-center gap-1.5">
          {mode === "geometry" && (
            <>
              <button
                type="button"
                onClick={() => setShowCameras((v) => !v)}
                className={`rounded-md px-2 py-1 text-[11px] font-medium ring-1 ${
                  showCameras
                    ? "bg-sky-600/80 text-white ring-sky-400/50"
                    : "bg-slate-900/80 text-slate-300 ring-slate-700"
                }`}
              >
                Cameras
              </button>
              <button
                type="button"
                onClick={() => setShowAxes((v) => !v)}
                className={`rounded-md px-2 py-1 text-[11px] font-medium ring-1 ${
                  showAxes
                    ? "bg-emerald-600/80 text-white ring-emerald-400/50"
                    : "bg-slate-900/80 text-slate-300 ring-slate-700"
                }`}
              >
                Ellipsoid Axes
              </button>
              <label className="flex items-center gap-1 rounded-md bg-slate-900/80 px-2 py-1 text-[11px] text-slate-300 ring-1 ring-slate-700">
                <span>Size</span>
                <input
                  type="range"
                  min={0.015}
                  max={0.12}
                  step={0.005}
                  value={pointSize}
                  onChange={(e) => setPointSize(Number(e.target.value))}
                  className="w-14"
                />
              </label>
            </>
          )}

          {mode === "mesh" && (
            <div className="flex rounded-md bg-slate-900/85 p-0.5 ring-1 ring-slate-700">
              {(["shaded", "both", "wireframe"] as const).map((wMode) => (
                <button
                  key={wMode}
                  type="button"
                  onClick={() => setMeshWireframe(wMode)}
                  className={`rounded px-2 py-0.5 text-[11px] font-medium capitalize ${
                    meshWireframe === wMode ? "bg-sky-600 text-white" : "text-slate-300 hover:text-white"
                  }`}
                >
                  {wMode === "both" ? "Shaded+Wire" : wMode}
                </button>
              ))}
            </div>
          )}

          {mode === "triangle_splats" && (
            <>
              <button
                type="button"
                onClick={() => setSoftTriSplats((v) => !v)}
                className={`rounded-md px-2 py-1 text-[11px] font-medium ring-1 ${
                  softTriSplats
                    ? "bg-sky-600/80 text-white ring-sky-400/50"
                    : "bg-slate-900/80 text-slate-300 ring-slate-700"
                }`}
              >
                {softTriSplats ? "Soft Triangle Splats" : "Crisp Surfels"}
              </button>
              <label className="flex items-center gap-1 rounded-md bg-slate-900/80 px-2 py-1 text-[11px] text-slate-300 ring-1 ring-slate-700">
                <span>Scale</span>
                <input
                  type="range"
                  min={0.5}
                  max={2.2}
                  step={0.1}
                  value={triScale}
                  onChange={(e) => setTriScale(Number(e.target.value))}
                  className="w-14"
                />
              </label>
            </>
          )}

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

