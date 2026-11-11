"""Integration tests for FastAPI backend endpoints and async job lifecycle."""

import time
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from backend.main import app
from tests.test_pipeline import _generate_textured_orbit_video


def test_health_and_project_upload_reconstruction_flow(tmp_path: Path):
    """Test /api/health, video upload, async job start, polling, and export endpoints."""
    video_path = tmp_path / "sample_scene.mp4"
    _generate_textured_orbit_video(video_path, num_frames=30, width=256, height=192)

    with TestClient(app) as client:
        # 1. Health check reports available CPU/GPU devices
        h = client.get("/api/health")
        assert h.status_code == 200
        hdata = h.json()
        assert hdata["status"] == "ok"
        assert hdata["device"] in ("cpu", "mps", "cuda")
        assert "cpu" in hdata["available_devices"]

        # 2. Upload video
        with video_path.open("rb") as f:
            up = client.post(
                "/api/upload",
                files={"file": ("sample_scene.mp4", f, "video/mp4")},
                data={"name": "API Integration Test", "description": "Textured orbit"},
            )
        assert up.status_code == 200, up.text
        proj = up.json()
        project_id = proj["id"]
        assert proj["video_width"] == 256
        assert proj["video_height"] == 192

        try:
            # 3. Start async reconstruction with fast test budget
            start = client.post(
                f"/api/reconstruction/{project_id}/start",
                json={
                    "max_resolution": 256,
                    "frame_interval": 3,
                    "max_keyframes": 8,
                    "max_gaussians": 500,
                    "iterations_per_batch": 20,
                    "final_iterations": 10,
                    "initial_batch_size": 5,
                    "incremental_batch_size": 3,
                    "cpu_threads": 4,
                    "sfm_backend": "auto",
                    "sh_degree": 0,
                    "adaptive_early_stop": False,
                    "resume": False,
                },
            )
            assert start.status_code == 200, start.text
            job_id = start.json()["id"]

            # 4. Poll until completed (timeout 120s)
            deadline = time.time() + 120
            job_data = None
            while time.time() < deadline:
                r = client.get(f"/api/reconstruction/jobs/{job_id}")
                assert r.status_code == 200
                job_data = r.json()
                if job_data["status"] in ("completed", "failed"):
                    break
                time.sleep(1.0)

            assert job_data is not None
            assert job_data["status"] == "completed", f"Job failed: {job_data.get('error_message')}"
            assert job_data["num_gaussians"] > 0
            assert job_data["has_ply"] is True
            assert job_data["has_splat"] is True

            # 5. Verify exports and incremental preview downloads
            ply_resp = client.get(f"/api/export/{job_id}/ply")
            assert ply_resp.status_code == 200
            assert len(ply_resp.content) > 100

            splat_resp = client.get(f"/api/export/{job_id}/splat")
            assert splat_resp.status_code == 200
            assert len(splat_resp.content) == job_data["num_gaussians"] * 32

            prev_resp = client.get(f"/api/export/{job_id}/preview/latest")
            assert prev_resp.status_code == 200

            bench_resp = client.get(f"/api/export/{job_id}/benchmarks")
            assert bench_resp.status_code == 200

            # 6. Verify derived representation endpoints (geometry, mesh, triangle_splats, mesh.obj)
            for rep_mode in ("geometry", "mesh", "triangle_splats"):
                rep = client.get(f"/api/export/{job_id}/representation/{rep_mode}")
                assert rep.status_code == 200, rep.text
                assert rep.json()["mode"] == rep_mode

            obj_resp = client.get(f"/api/export/{job_id}/mesh")
            assert obj_resp.status_code == 200
        finally:
            # Clean up test project from DB & disk
            client.delete(f"/api/projects/{project_id}")


def test_photo_set_upload_and_reconstruction_flow(tmp_path: Path):
    """Test /api/upload/photos with multiple images -> incremental 3DGS -> export."""
    video_path = tmp_path / "src.mp4"
    _generate_textured_orbit_video(video_path, num_frames=12, width=256, height=192)

    # Extract 8 photo files as JPEG/PNG
    cap = cv2.VideoCapture(str(video_path))
    photo_paths: list[Path] = []
    idx = 0
    while len(photo_paths) < 8:
        ok, frame = cap.read()
        if not ok:
            break
        ext = ".jpg" if idx % 2 == 0 else ".png"
        p = tmp_path / f"view_{idx:02d}{ext}"
        cv2.imwrite(str(p), frame)
        photo_paths.append(p)
        idx += 1
    cap.release()
    assert len(photo_paths) == 8

    with TestClient(app) as client:
        opened = [p.open("rb") for p in photo_paths]
        try:
            files_payload = [
                ("files", (p.name, fh, "image/jpeg" if p.suffix == ".jpg" else "image/png"))
                for p, fh in zip(photo_paths, opened)
            ]
            up = client.post(
                "/api/upload/photos",
                files=files_payload,
                data={"name": "Photo Set Test", "description": "8 multi-view photos"},
            )
        finally:
            for fh in opened:
                fh.close()

        assert up.status_code == 200, up.text
        proj = up.json()
        project_id = proj["id"]
        assert proj["video_filename"] == "8 photos"
        assert proj["video_width"] == 256
        assert proj["video_height"] == 192

        try:
            start = client.post(
                f"/api/reconstruction/{project_id}/start",
                json={
                    "max_resolution": 256,
                    "frame_interval": 1,
                    "max_keyframes": 8,
                    "max_gaussians": 500,
                    "iterations_per_batch": 20,
                    "final_iterations": 10,
                    "initial_batch_size": 5,
                    "incremental_batch_size": 3,
                    "cpu_threads": 4,
                    "sfm_backend": "auto",
                    "sh_degree": 0,
                    "adaptive_early_stop": False,
                    "resume": False,
                },
            )
            assert start.status_code == 200, start.text
            job_id = start.json()["id"]

            deadline = time.time() + 120
            job_data = None
            while time.time() < deadline:
                r = client.get(f"/api/reconstruction/jobs/{job_id}")
                assert r.status_code == 200
                job_data = r.json()
                if job_data["status"] in ("completed", "failed"):
                    break
                time.sleep(1.0)

            assert job_data is not None
            assert job_data["status"] == "completed", f"Photo job failed: {job_data.get('error_message')}"
            assert job_data["num_gaussians"] > 0
            assert job_data["has_ply"] is True
            assert job_data["has_splat"] is True
        finally:
            client.delete(f"/api/projects/{project_id}")

