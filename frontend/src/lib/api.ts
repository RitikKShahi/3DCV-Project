import type {
  GeometryRepresentation,
  HealthResponse,
  JobResponse,
  MeshRepresentation,
  ProjectResponse,
  ReconstructionParams,
  TriangleSplatsRepresentation,
} from "./types";

const API_BASE = "";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, init);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        detail = body.detail
          .map((item: { loc?: unknown[]; msg?: string }) => {
            const loc = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
            return loc ? `${loc}: ${item.msg || "Invalid value"}` : item.msg || JSON.stringify(item);
          })
          .join("; ");
      } else if (body.detail && typeof body.detail === "object") {
        detail = JSON.stringify(body.detail);
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export async function listProjects(): Promise<ProjectResponse[]> {
  return request<ProjectResponse[]>("/api/projects");
}

export async function getProject(projectId: string): Promise<ProjectResponse> {
  return request<ProjectResponse>(`/api/projects/${projectId}`);
}

export async function deleteProject(projectId: string): Promise<void> {
  await request(`/api/projects/${projectId}`, { method: "DELETE" });
}

export async function uploadVideo(
  file: File,
  name: string,
  description: string
): Promise<ProjectResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("name", name);
  form.append("description", description);
  return request<ProjectResponse>("/api/upload", {
    method: "POST",
    body: form,
  });
}

export async function uploadPhotos(
  files: File[],
  name: string,
  description: string
): Promise<ProjectResponse> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  form.append("name", name);
  form.append("description", description);
  return request<ProjectResponse>("/api/upload/photos", {
    method: "POST",
    body: form,
  });
}

export async function startReconstruction(
  projectId: string,
  params: ReconstructionParams
): Promise<JobResponse> {
  return request<JobResponse>(`/api/reconstruction/${projectId}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function getJobStatus(jobId: string): Promise<JobResponse> {
  return request<JobResponse>(`/api/reconstruction/jobs/${jobId}`);
}

export function getExportUrl(jobId: string, format: "ply" | "splat" | "benchmarks" | "mesh"): string {
  return `/api/export/${jobId}/${format}`;
}

export async function fetchRepresentation(
  jobId: string,
  mode: "geometry" | "mesh" | "triangle_splats"
): Promise<GeometryRepresentation | MeshRepresentation | TriangleSplatsRepresentation> {
  return request(`/api/export/${jobId}/representation/${mode}`);
}

export function getPreviewSplatUrl(jobId: string, batchIdx?: number): string {
  if (typeof batchIdx === "number") {
    return `/api/export/${jobId}/preview/batch/${batchIdx}`;
  }
  return `/api/export/${jobId}/preview/latest`;
}

export function getBatchPreviewImageUrl(jobId: string, batchIdx: number): string {
  return `/api/export/${jobId}/preview/image/${batchIdx}`;
}

export function getWsUrl(jobId: string): string {
  if (typeof window === "undefined") return "";
  const backendOrigin =
    process.env.NEXT_PUBLIC_WS_URL ||
    `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000`;
  return `${backendOrigin}/ws/jobs/${jobId}`;
}

