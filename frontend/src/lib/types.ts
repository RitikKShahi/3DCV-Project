export interface ReconstructionParams {
  max_resolution: number;
  frame_interval: number;
  max_keyframes: number;
  max_gaussians: number;
  iterations_per_batch: number;
  final_iterations: number;
  initial_batch_size: number;
  incremental_batch_size: number;
  cpu_threads: number;
  device: "auto" | "cpu" | "mps" | "cuda";
  sfm_backend: "auto" | "pycolmap" | "opencv";
  sh_degree: number;
  camera_fov_deg: number;
  adaptive_early_stop: boolean;
  resume: boolean;
}

export interface HealthResponse {
  status: string;
  device: string;
  recommended_device?: string;
  available_devices?: string[];
  cuda_available: boolean;
  mps_available?: boolean;
  mps_detected?: boolean;
  mps_note?: string;
  torch_version: string;
  cpu_cores: number;
}

export interface BatchMetric {
  batch_index: number;
  num_cameras: number;
  new_cameras: number;
  num_gaussians: number;
  gaussians_added: number;
  sfm_backend: string;
  sfm_reproj_error: number;
  final_loss: number;
  mean_psnr: number;
  iterations_run: number;
  elapsed_sec: number;
  preview_splat?: string;
  preview_image?: string | null;
}

export interface JobResponse {
  id: string;
  project_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  progress: number;
  message: string;
  batch_index: number;
  total_batches: number;
  num_keyframes: number;
  num_cameras: number;
  num_gaussians: number;
  psnr: number;
  loss: number;
  sfm_backend: string;
  elapsed_sec: number;
  peak_memory_mb: number;
  has_ply: boolean;
  has_splat: boolean;
  has_preview: boolean;
  preview_batches: number[];
  metrics_history: BatchMetric[];
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectResponse {
  id: string;
  name: string;
  description: string;
  video_filename: string;
  video_width: number;
  video_height: number;
  video_duration_sec: number;
  status: string;
  created_at: string;
  updated_at: string;
  latest_job: JobResponse | null;
}

export interface ProgressWsMessage {
  job_id: string;
  project_id: string;
  status: "running" | "completed" | "failed";
  stage: string;
  progress: number;
  message: string;
  batch_index?: number;
  total_batches?: number;
  num_gaussians?: number;
  num_cameras?: number;
  psnr?: number;
  loss?: number;
  elapsed_sec?: number;
  peak_memory_mb?: number;
  preview_url?: string | null;
  has_new_preview?: boolean;
  error_message?: string;
}

export type PreviewRenderMode = "gaussian" | "geometry" | "mesh" | "triangle_splats";

export interface GeometryCamera {
  name: string;
  center: [number, number, number];
  forward: [number, number, number];
  up: [number, number, number];
  right: [number, number, number];
}

export interface GeometryRepresentation {
  mode: "geometry";
  count: number;
  positions: number[];
  colors: number[];
  scales: number[];
  rotations: number[];
  opacities: number[];
  cameras: GeometryCamera[];
}

export interface MeshRepresentation {
  mode: "mesh";
  vertex_count: number;
  face_count: number;
  vertices: number[];
  faces: number[];
  normals: number[];
  colors: number[];
}

export interface TriangleSplatsRepresentation {
  mode: "triangle_splats";
  triangle_count: number;
  positions: number[];
  normals: number[];
  colors: number[];
  opacities: number[];
  uvs: number[];
}

