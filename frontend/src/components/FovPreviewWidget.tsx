"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Eye } from "lucide-react";

interface FovPreviewWidgetProps {
  fovDeg: number; // 0 = Auto, or 25..120
  onChange: (fovDeg: number) => void;
  previewImageUrl?: string | null;
}

const PRESETS = [
  { label: "35° Tele", val: 35 },
  { label: "50° Normal", val: 50 },
  { label: "65° Wide", val: 65 },
  { label: "85° Ultra", val: 85 },
  { label: "105° Fish", val: 105 },
];

function eqFocalMm(deg: number): number {
  const rad = (Math.max(20, Math.min(130, deg)) * Math.PI) / 180;
  return Math.round(18 / Math.tan(rad / 2));
}

export default function FovPreviewWidget({
  fovDeg,
  onChange,
  previewImageUrl,
}: FovPreviewWidgetProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const camRef = useRef<THREE.PerspectiveCamera | null>(null);
  const [customActive, setCustomActive] = useState(fovDeg > 0);
  const effectiveFov = fovDeg > 0 ? fovDeg : 60;

  useEffect(() => {
    setCustomActive(fovDeg > 0);
  }, [fovDeg]);

  // Live 3D WebGL perspective preview responding immediately to FOV slider
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const w = el.clientWidth || 300;
    const h = el.clientHeight || 160;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x050914, 1);
    el.innerHTML = "";
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(effectiveFov, w / h, 0.1, 100);
    camera.position.set(0, 1.35, 3.8);
    camera.lookAt(0, 0.45, 0);
    camRef.current = camera;

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dl = new THREE.DirectionalLight(0x7dd3fc, 1.1);
    dl.position.set(3, 5, 4);
    scene.add(dl);

    // Reference ground grid + sample 3D subject & surrounding pillars to visualize FOV framing
    const grid = new THREE.GridHelper(10, 12, 0x1e293b, 0x0f172a);
    scene.add(grid);

    const subject = new THREE.Mesh(
      new THREE.BoxGeometry(0.9, 0.9, 0.9),
      new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.4, metalness: 0.1 })
    );
    subject.position.set(0, 0.45, 0);
    scene.add(subject);

    const leftPillar = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22, 0.22, 1.4, 12),
      new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.5 })
    );
    leftPillar.position.set(-2.0, 0.7, -0.8);
    scene.add(leftPillar);

    const rightPillar = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22, 0.22, 1.4, 12),
      new THREE.MeshStandardMaterial({ color: 0xa78bfa, roughness: 0.5 })
    );
    rightPillar.position.set(2.0, 0.7, -0.8);
    scene.add(rightPillar);

    let raf = 0;
    const renderLoop = () => {
      subject.rotation.y += 0.006;
      renderer.render(scene, camera);
      raf = requestAnimationFrame(renderLoop);
    };
    renderLoop();

    return () => {
      cancelAnimationFrame(raf);
      camRef.current = null;
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    if (camRef.current) {
      camRef.current.fov = effectiveFov;
      camRef.current.updateProjectionMatrix();
    }
  }, [effectiveFov]);

  // Top-down SVG frustum cone geometry
  const halfAngleRad = ((Math.max(20, Math.min(120, effectiveFov)) / 2) * Math.PI) / 180;
  const coneLen = 78;
  const dx = Math.sin(halfAngleRad) * coneLen;
  const dy = Math.cos(halfAngleRad) * coneLen;
  const ox = 90;
  const oy = 100;

  return (
    <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/70 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Eye className="h-4 w-4 text-sky-400" />
          <span className="text-xs font-semibold text-white">Camera Field of View (FOV) & Live Preview</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              setCustomActive(false);
              onChange(0);
            }}
            className={`rounded-md px-2.5 py-1 text-xs font-medium ${
              !customActive ? "bg-sky-500 text-white" : "bg-slate-800 text-slate-300 hover:text-white"
            }`}
          >
            Auto-Detect (COLMAP)
          </button>
          <button
            type="button"
            onClick={() => {
              setCustomActive(true);
              if (fovDeg <= 0) onChange(65);
            }}
            className={`rounded-md px-2.5 py-1 text-xs font-medium ${
              customActive ? "bg-sky-500 text-white" : "bg-slate-800 text-slate-300 hover:text-white"
            }`}
          >
            Custom FOV
          </button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <input
          type="range"
          min={25}
          max={120}
          step={1}
          value={effectiveFov}
          onChange={(e) => {
            setCustomActive(true);
            onChange(Number(e.target.value));
          }}
          className="flex-1 min-w-[160px]"
        />
        <div className="flex items-center gap-1.5">
          <input
            type="number"
            min={25}
            max={120}
            value={Math.round(effectiveFov)}
            onChange={(e) => {
              const v = Math.max(25, Math.min(120, Number(e.target.value) || 60));
              setCustomActive(true);
              onChange(v);
            }}
            className="w-16 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-center font-mono text-xs text-white"
          />
          <span className="text-xs text-slate-400">° (~{eqFocalMm(effectiveFov)}mm eq.)</span>
        </div>
      </div>

      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <button
            key={p.val}
            type="button"
            onClick={() => {
              setCustomActive(true);
              onChange(p.val);
            }}
            className={`rounded-md px-2 py-1 text-[11px] font-medium transition ${
              customActive && Math.round(effectiveFov) === p.val
                ? "bg-sky-600 text-white"
                : "bg-slate-800/80 text-slate-300 hover:bg-slate-800"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {/* Top-down frustum cone diagram */}
        <div className="flex flex-col items-center justify-center rounded-lg border border-slate-800 bg-slate-900/50 p-2">
          <svg viewBox="0 0 180 115" className="h-32 w-full max-w-[220px]">
            <polygon
              points={`${ox},${oy} ${ox - dx},${oy - dy} ${ox + dx},${oy - dy}`}
              fill="rgba(56, 189, 248, 0.16)"
              stroke="#38bdf8"
              strokeWidth="1.5"
            />
            <circle cx={ox - 32} cy={35} r="5" fill="#10b981" opacity="0.85" />
            <rect x={ox - 8} y={38} width="16" height="16" rx="2" fill="#38bdf8" />
            <circle cx={ox + 32} cy={35} r="5" fill="#a78bfa" opacity="0.85" />
            <circle cx={ox} cy={oy} r="4" fill="#f8fafc" />
            <text x={ox} y={oy + 12} textAnchor="middle" fill="#94a3b8" fontSize="9">
              {customActive ? `Camera FOV: ${Math.round(effectiveFov)}°` : "Auto (~60° preview)"}
            </text>
          </svg>
        </div>

        {/* Live 3D perspective preview + optional uploaded media thumbnail overlay */}
        <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950 h-36">
          <div ref={mountRef} className="h-full w-full" />
          {previewImageUrl && (
            <img
              src={previewImageUrl}
              alt="Input preview"
              className="pointer-events-none absolute right-2 bottom-2 h-12 w-16 rounded border border-slate-700 object-cover opacity-85 shadow"
            />
          )}
          <span className="pointer-events-none absolute left-2 top-2 rounded bg-slate-900/80 px-2 py-0.5 text-[10px] text-sky-300 ring-1 ring-slate-700">
            Live {Math.round(effectiveFov)}° Framing Preview
          </span>
        </div>
      </div>
    </div>
  );
}

