"""Local Seal web app API. Run with: python web_server.py"""
import asyncio
import csv
import json
import math
import os
import shutil
import threading
import time
import uuid
import zipfile
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen
from io import BytesIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.requests import Request
from ultralytics import YOLO
import uvicorn

from inference_engine import (resolve_model_path, run_tracking, run_background_check, save_csv,
                             save_deliverable_bundle, track_color)
from species_engine import classify_tracks
from video_encoding import preferred_video_suffix
from tracking_metrics import evaluate_tracking

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
DEFAULT_WEIGHTS = Path(os.environ.get("SEAL_DETECTOR_WEIGHTS", APP_DIR / "weights" / "detector_v26_n3.zip"))
RUNS_DIR = APP_DIR / "exports" / "web_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}
occurrence_cache: dict[str, tuple[float, dict]] = {}
occurrence_cache_lock = threading.Lock()
jobs_lock = threading.Lock()
model_lock = threading.Lock()
model_cache: dict[str, YOLO] = {}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seal-inference")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
video_conversion_lock = threading.Lock()
LIVE_DEMO_VIDEO = Path(os.environ["SEAL_LIVE_DEMO_VIDEO"]) if os.environ.get("SEAL_LIVE_DEMO_VIDEO") else APP_DIR / "__no_live_demo_video__"
live_sessions: dict[str, dict] = {}
live_lock = threading.Lock()
detector_execution_lock = threading.Lock()


def cached_model(model_path: Path) -> YOLO:
    key = str(model_path.resolve())
    with model_lock:
        if key not in model_cache:
            model_cache[key] = YOLO(key, task="detect")
        return model_cache[key]


def set_job(job_id: str, **values):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(values)


def process_job(job_id: str, workdir: Path, video_path: Path, confidence: float, enhancement: str, imgsz: int,
                background_path: Path | None = None, species_enabled: bool = False):
    job_started = time.perf_counter()
    try:
        set_job(job_id, state="loading", message="Loading detector…")
        if not DEFAULT_WEIGHTS.exists():
            raise FileNotFoundError("The bundled Fishial detector v26 n3 is missing from weights/detector_v26_n3.zip.")
        model_source = DEFAULT_WEIGHTS
        model_dest = APP_DIR / ".web_models"
        model_dest.mkdir(exist_ok=True)
        try:
            model_path = resolve_model_path(model_source, model_dest)
            model = cached_model(model_path)
        except Exception as exc:
            if isinstance(exc, (ValueError, FileNotFoundError)):
                raise
            raise ValueError("Could not load this checkpoint. Use an Ultralytics YOLO detection model (.pt) or a ZIP containing one.") from exc

        output_path = workdir / f"seal_annotated{preferred_video_suffix()}"
        species_crops = defaultdict(list) if species_enabled else None
        with detector_execution_lock:
            rows, fields, metrics = run_tracking(
                model, video_path, output_path, confidence,
                progress_callback=lambda done, total: set_job(job_id, state="processing", progress=done / total if total else 0,
                    processed_frames=done, total_frames=total, message=f"Detecting and tracking · {done:,} / {total or '?'} frames"),
                enhancement=enhancement, imgsz=imgsz,
                preview_callback=lambda frame, done: save_live_preview(job_id, workdir, frame, done),
                render_output=not species_enabled, crop_candidates=species_crops,
            )
        metrics["annotated_video_extension"] = output_path.suffix.lower()
        metrics["model_file"] = "Fishial detector v26 n3 · detector_v26_n3.zip"
        metrics["model_size_mb"] = round(DEFAULT_WEIGHTS.stat().st_size / (1024 * 1024), 2)
        if species_enabled:
            set_job(job_id, state="classifying", progress=0, processed_frames=0, total_frames=0,
                    message="Classifying sampled crops for each tracked fish…")
            species_output = workdir / f"seal_species_annotated{output_path.suffix}"
            species_started = time.perf_counter()
            metrics["species_classification"] = classify_tracks(
                video_path, rows, species_output,
                crop_candidates=species_crops,
                progress_callback=lambda done, total: set_job(
                    job_id, progress=done / total if total else 1,
                    processed_frames=done, total_frames=total,
                    message=f"Classifying fish tracks · {done:,} / {total or 0:,} tracks",
                ),
            )
            metrics["species_classification_seconds"] = round(time.perf_counter() - species_started, 3)
            metrics["video_encoder"] = metrics["species_classification"].get("video_encoder", "unknown")
            # Match Streamlit: its species pass re-renders the source clip with the same boxes and labels.
            shutil.move(str(species_output), str(output_path))
            fields = fields + ["species_label", "species_score", "species_unlabelled", "species_agreement", "species_top3"]
        csv_path = workdir / "seal_tracks.csv"
        save_csv(rows, fields, csv_path)
        if background_path is not None:
            set_job(job_id, message="Checking detections on the fish-free background clip…")
            with detector_execution_lock:
                metrics["background_video_check"] = run_background_check(model, background_path, confidence, enhancement, imgsz)
        metrics["total_elapsed_seconds"] = round(time.perf_counter() - job_started, 3)
        with jobs_lock:
            metrics["source_filename"] = jobs.get(job_id, {}).get("source_filename", "Uploaded video")
        metrics_path = workdir / "seal_benchmark.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        artifacts = save_deliverable_bundle(rows, metrics, workdir)
        static_video_dir = WEB_DIR / "result_videos"
        static_video_dir.mkdir(exist_ok=True)
        shutil.copy2(output_path, static_video_dir / f"{job_id}{output_path.suffix.lower()}")
        (WEB_DIR / "latest.json").write_text(json.dumps({"job_id": job_id}), encoding="utf-8")
        set_job(job_id, state="done", progress=1, message="Analysis complete", metrics=metrics,
                video_url=f"/api/jobs/{job_id}/video", csv_url=f"/api/jobs/{job_id}/tracks.csv",
                benchmark_url=f"/api/jobs/{job_id}/benchmark.json",
                bundle_url=f"/api/jobs/{job_id}/bundle.zip",
                species_enabled=species_enabled,
                video_extension=output_path.suffix.lower(),
                mot_url=f"/api/jobs/{job_id}/mot.txt", centroids_url=f"/api/jobs/{job_id}/centroids.csv",
                counts_url=f"/api/jobs/{job_id}/counts.csv", heatmap_url=f"/api/jobs/{job_id}/heatmap.png",
                trajectories_url=f"/api/jobs/{job_id}/trajectories.png")
    except Exception as exc:
        set_job(job_id, state="error", message=str(exc))


async def create_job(request: Request):
    try:
        with live_lock:
            if any(session.get("state") in {"loading", "live"} for session in live_sessions.values()):
                return JSONResponse({"error": "Stop Live Monitoring before starting an uploaded-video analysis."}, status_code=409)
        form = await request.form()
        video = form.get("video")
        model_file = form.get("model")
        background = form.get("background")
        confidence_raw = form.get("confidence", "0.40")
        if not isinstance(video, UploadFile) or not video.filename:
            return JSONResponse({"error": "Choose an underwater video first."}, status_code=400)
        video_ext = Path(video.filename).suffix.lower()
        if video_ext not in VIDEO_EXTENSIONS:
            return JSONResponse({"error": "Video must be MP4, MOV, AVI, or MKV."}, status_code=400)
        if model_file and isinstance(model_file, UploadFile) and model_file.filename:
            return JSONResponse({"error": "This benchmark is locked to the bundled Fishial detector v26 n3 so runs remain comparable."}, status_code=400)
        if background and (not isinstance(background, UploadFile) or not background.filename):
            background = None
        if background and Path(background.filename).suffix.lower() not in VIDEO_EXTENSIONS:
            return JSONResponse({"error": "Fish-free background clip must be MP4, MOV, AVI, or MKV."}, status_code=400)
        confidence = float(confidence_raw)
        if not 0.05 <= confidence <= 0.9:
            return JSONResponse({"error": "Confidence must be between 0.05 and 0.90."}, status_code=400)
        enhancement = str(form.get("enhancement", "original"))
        if enhancement not in {"original", "clahe"}:
            return JSONResponse({"error": "Choose Original frames or CLAHE contrast enhancement."}, status_code=400)
        imgsz = int(form.get("imgsz", "640"))
        if imgsz not in {640, 960, 1280}:
            return JSONResponse({"error": "Inference image size must be 640, 960, or 1280."}, status_code=400)
        species_enabled = str(form.get("species", "false")).lower() in {"1", "true", "yes", "on"}

        job_id = uuid.uuid4().hex
        workdir = RUNS_DIR / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        video_path = workdir / f"input{video_ext}"
        video.file.seek(0)
        with video_path.open("wb") as destination:
            shutil.copyfileobj(video.file, destination, length=1024 * 1024)
        background_path = None
        if background:
            background_path = workdir / f"background{Path(background.filename).suffix.lower()}"
            background.file.seek(0)
            with background_path.open("wb") as destination:
                shutil.copyfileobj(background.file, destination, length=1024 * 1024)
        with jobs_lock:
            jobs[job_id] = {"state": "queued", "progress": 0, "processed_frames": 0, "total_frames": 0,
                            "message": "Queued for analysis", "source_filename": Path(video.filename).name}
        executor.submit(process_job, job_id, workdir, video_path, confidence, enhancement, imgsz, background_path, species_enabled)
        return JSONResponse({"job_id": job_id})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Could not start analysis: {exc}"}, status_code=500)


async def job_status(request: Request):
    job_id = request.path_params["job_id"]
    with jobs_lock:
        job = jobs.get(job_id)
        result = dict(job) if job else None
    if result is None:
        workdir = RUNS_DIR / job_id
        metrics_path = workdir / "seal_benchmark.json"
        video_path = workdir / "seal_annotated.webm"
        legacy_video_path = workdir / "seal_annotated.mp4"
        csv_path = workdir / "seal_tracks.csv"
        if metrics_path.is_file() and csv_path.is_file() and (video_path.is_file() or legacy_video_path.is_file()):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            result = {"state": "done", "progress": 1, "processed_frames": metrics.get("frames_processed", 0),
                "total_frames": metrics.get("frames_processed", 0), "message": "Analysis complete", "metrics": metrics,
                "video_url": f"/api/jobs/{job_id}/video", "csv_url": f"/api/jobs/{job_id}/tracks.csv",
                "benchmark_url": f"/api/jobs/{job_id}/benchmark.json", "bundle_url": f"/api/jobs/{job_id}/bundle.zip",
                "heatmap_url": f"/api/jobs/{job_id}/heatmap.png", "trajectories_url": f"/api/jobs/{job_id}/trajectories.png",
                "species_enabled": bool(metrics.get("species_classification")),
                "video_extension": metrics.get("annotated_video_extension", ".webm")}
    return JSONResponse(result or {"error": "Analysis session not found. Restart a new analysis."}, status_code=200 if result else 404)


async def latest_job(request: Request):
    completed = sorted((p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "seal_benchmark.json").is_file()),
                       key=lambda p: (p / "seal_benchmark.json").stat().st_mtime, reverse=True)
    for workdir in completed:
        job_id = workdir.name
        has_video = (workdir / "seal_annotated.webm").is_file() or (workdir / "seal_annotated.mp4").is_file()
        if not has_video or not (workdir / "seal_tracks.csv").is_file():
            continue
        try:
            metrics = json.loads((workdir / "seal_benchmark.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return JSONResponse({"job_id": job_id, "state": "done", "progress": 1,
            "processed_frames": metrics.get("frames_processed", 0), "total_frames": metrics.get("frames_processed", 0),
            "message": "Restored latest completed analysis", "metrics": metrics,
            "video_url": f"/api/jobs/{job_id}/video", "csv_url": f"/api/jobs/{job_id}/tracks.csv",
            "benchmark_url": f"/api/jobs/{job_id}/benchmark.json", "bundle_url": f"/api/jobs/{job_id}/bundle.zip",
            "heatmap_url": f"/api/jobs/{job_id}/heatmap.png", "trajectories_url": f"/api/jobs/{job_id}/trajectories.png",
            "species_enabled": bool(metrics.get("species_classification")),
            "video_extension": metrics.get("annotated_video_extension", ".webm")})
    return JSONResponse({"error": "No completed video analysis is available yet."}, status_code=404)


def save_live_preview(job_id: str, workdir: Path, frame, frame_number: int):
    preview_path = workdir / "live_preview.jpg"
    if not cv2.imwrite(str(preview_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 82]):
        return
    set_job(job_id, preview_url=f"/api/jobs/{job_id}/preview.jpg", preview_frame=frame_number)


def persist_live_snapshot(session_id: str, session: dict):
    keys = ("state", "video_path", "frame_number", "loop_count", "detection_count", "track_count",
            "processed_frames", "inference_ms", "mean_inference_ms", "processing_fps", "elapsed_seconds",
            "unique_track_ids_this_loop", "source_fps", "frame_width", "frame_height", "hardware",
            "count_history", "centroid_history", "error")
    with live_lock:
        snapshot = {key: session.get(key) for key in keys}
    folder = RUNS_DIR / "live" / session_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "live_stats.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def run_live_loop(session_id: str, video_path: Path, confidence: float, imgsz: int):
    """Loop a local clip through the bundled detector and publish annotated JPEG frames."""
    with live_lock:
        session = live_sessions.get(session_id)
    if session is None:
        return
    cap = None
    try:
        if not DEFAULT_WEIGHTS.exists():
            raise FileNotFoundError("The bundled Fishial detector is missing from weights/detector_v26_n3.zip.")
        model_path = resolve_model_path(DEFAULT_WEIGHTS, APP_DIR / ".web_models")
        model = cached_model(model_path)
        with detector_execution_lock:
            for tracker in getattr(getattr(model, "predictor", None), "trackers", []) or []:
                if hasattr(tracker, "reset"):
                    tracker.reset()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open this clip. Try exporting it as MP4 (H.264).")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, round(source_fps / 15.0))
        frame_idx = 0
        loop_count = 1
        from collections import defaultdict, deque
        trails = defaultdict(lambda: deque(maxlen=40))
        last_seen = {}
        loop_track_ids = set()
        try:
            model_device = next(model.model.parameters()).device
            if getattr(model_device, "type", "cpu") == "cuda":
                import torch
                device_label = f"GPU · {torch.cuda.get_device_name(model_device)}"
            else:
                device_label = "CPU"
        except Exception:
            device_label = "Automatic · device unavailable"
        with live_lock:
            session.update({"state": "live", "source_fps": round(source_fps, 2), "loop_count": loop_count,
                            "frame_width": source_width, "frame_height": source_height, "hardware": device_label})
        while not session["stop"].is_set():
            ok, frame = cap.read()
            if not ok:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    raise ValueError("Could not rewind the clip for its next loop.")
                frame_idx = 0
                loop_count += 1
                trails.clear()
                last_seen.clear()
                loop_track_ids.clear()
                with detector_execution_lock:
                    for tracker in getattr(getattr(model, "predictor", None), "trackers", []) or []:
                        if hasattr(tracker, "reset"):
                            tracker.reset()
                with live_lock:
                    session["loop_count"] = loop_count
                continue
            current_frame = frame_idx
            frame_idx += 1
            if (current_frame % stride) != 0:
                continue
            inference_started = time.perf_counter()
            with detector_execution_lock:
                result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=confidence,
                                     imgsz=imgsz, verbose=False)[0]
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            boxes = result.boxes
            detections = []
            xyxy, scores, track_ids = [], [], []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for box, score, track_id in zip(xyxy, scores, track_ids):
                    track_id = int(track_id) if track_id is not None else None
                    detections.append((box, float(score), track_id))
                    if track_id is not None:
                        loop_track_ids.add(track_id)
                        x1, y1, x2, y2 = box
                        trails[track_id].append((round((x1 + x2) / 2), round((y1 + y2) / 2)))
                        last_seen[track_id] = current_frame
            # Release Ultralytics result tensors before painting. Draw directly onto the
            # decoded frame to avoid several full-resolution copies on memory-constrained PCs.
            del boxes, result, xyxy, scores, track_ids
            for track_id, points in trails.items():
                if current_frame - last_seen.get(track_id, current_frame) > 6:
                    continue
                color = track_color(track_id)
                if len(points) > 1:
                    cv2.polylines(frame, [np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))],
                                  False, color, 2, cv2.LINE_AA)
                if points:
                    cv2.circle(frame, points[-1], 4, color, -1, cv2.LINE_AA)
            for box, score, track_id in detections:
                x1, y1, x2, y2 = (int(round(value)) for value in box)
                color = track_color(track_id) if track_id is not None else (0, 220, 180)
                label = f"ID {track_id} · {score:.2f}" if track_id is not None else f"Fish · {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.putText(frame, label, (max(0, x1), max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"SIMULATED LIVE  |  LOOP {loop_count}  |  FRAME {current_frame + 1}",
                        (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 250, 255), 2, cv2.LINE_AA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 76])
            if ok:
                frame_bytes = encoded.tobytes()
                del encoded
                with live_lock:
                    processed_frames = int(session.get("processed_frames", 0)) + 1
                    started = float(session.get("started_monotonic", inference_started))
                    elapsed = max(0.001, time.perf_counter() - started)
                    count_history = session.setdefault("count_history", [])
                    count_history.append({"t": round(elapsed, 2), "detections": len(detections),
                                          "tracks": len({item[2] for item in detections if item[2] is not None})})
                    if len(count_history) > 180:
                        del count_history[:-180]
                    centroid_history = session.setdefault("centroid_history", [])
                    for box, _score, track_id in detections:
                        x1, y1, x2, y2 = box
                        centroid_history.append({"x": round(((x1 + x2) / 2) / max(source_width, 1), 4),
                                                 "y": round(((y1 + y2) / 2) / max(source_height, 1), 4),
                                                 "track_id": track_id, "frame": current_frame + 1,
                                                 "loop": loop_count})
                    if len(centroid_history) > 600:
                        del centroid_history[:-600]
                    inference_total_ms = float(session.get("inference_total_ms", 0.0)) + inference_ms
                    session.update({"frame": frame_bytes, "frame_number": current_frame + 1,
                                    "loop_count": loop_count, "detection_count": len(detections),
                                    "track_count": len({item[2] for item in detections if item[2] is not None}),
                                    "width": source_width, "processed_frames": processed_frames,
                                    "inference_ms": round(inference_ms, 2),
                                    "mean_inference_ms": round(inference_total_ms / processed_frames, 2),
                                    "inference_total_ms": inference_total_ms,
                                    "processing_fps": round(processed_frames / elapsed, 2),
                                    "unique_track_ids_this_loop": len(loop_track_ids),
                                    "elapsed_seconds": round(elapsed, 1)})
            session["stop"].wait(max(0.0, (1.0 / 15.0) - (time.perf_counter() - inference_started)))
    except Exception as exc:
        with live_lock:
            session["state"] = "error"
            session["error"] = str(exc)
    finally:
        if cap is not None:
            cap.release()
        with live_lock:
            if session.get("state") != "error":
                session["state"] = "stopped"
        persist_live_snapshot(session_id, session)


async def live_demo_info(request: Request):
    return JSONResponse({"available": LIVE_DEMO_VIDEO.is_file(), "filename": LIVE_DEMO_VIDEO.name})


async def active_live(request: Request):
    """Return the current feed so a refreshed Live page can restore its controls."""
    with live_lock:
        active = next(((session_id, session) for session_id, session in reversed(list(live_sessions.items()))
                       if session.get("state") in {"loading", "live"}), None)
        if active is None:
            return JSONResponse({"active": False})
        session_id, session = active
        return JSONResponse({"active": True, "session_id": session_id,
                             "stream_url": f"/api/live/{session_id}/stream",
                             "state": session.get("state")})


async def start_live(request: Request):
    try:
        with live_lock:
            if any(session.get("state") in {"loading", "live", "stopping"} for session in live_sessions.values()):
                return JSONResponse({"error": "A simulated live feed is already running. Stop it before starting another."}, status_code=409)
        with jobs_lock:
            if any(job.get("state") in {"queued", "loading", "processing", "classifying"} for job in jobs.values()):
                return JSONResponse({"error": "Wait for the current video analysis to finish before starting Live Monitoring."}, status_code=409)
        form_data = await request.form()
        uploaded = form_data.get("video")
        confidence = float(form_data.get("confidence", "0.40"))
        imgsz = int(form_data.get("imgsz", "640"))
        if not 0.05 <= confidence <= 0.90 or imgsz not in {640, 960, 1280}:
            return JSONResponse({"error": "Invalid confidence or image size."}, status_code=400)
        if isinstance(uploaded, UploadFile) and uploaded.filename:
            ext = Path(uploaded.filename).suffix.lower()
            if ext not in VIDEO_EXTENSIONS:
                return JSONResponse({"error": "Video must be MP4, MOV, AVI, or MKV."}, status_code=400)
            session_id = uuid.uuid4().hex
            session_dir = RUNS_DIR / "live" / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            video_path = session_dir / f"source{ext}"
            uploaded.file.seek(0)
            with video_path.open("wb") as destination:
                shutil.copyfileobj(uploaded.file, destination, length=1024 * 1024)
        elif LIVE_DEMO_VIDEO.is_file():
            session_id = uuid.uuid4().hex
            video_path = LIVE_DEMO_VIDEO
        else:
            return JSONResponse({"error": "Choose a video clip. The bundled demo clip is not present on this computer."}, status_code=400)
        (RUNS_DIR / "live" / session_id).mkdir(parents=True, exist_ok=True)
        session = {"state": "loading", "stop": threading.Event(), "frame": None, "frame_number": 0,
                   "loop_count": 1, "detection_count": 0, "track_count": 0, "video_path": str(video_path),
                   "started_monotonic": time.perf_counter(), "processed_frames": 0, "inference_total_ms": 0.0,
                   "count_history": [], "centroid_history": []}
        with live_lock:
            live_sessions[session_id] = session
        threading.Thread(target=run_live_loop, args=(session_id, video_path, confidence, imgsz),
                         name=f"seal-live-{session_id[:8]}", daemon=True).start()
        return JSONResponse({"session_id": session_id, "stream_url": f"/api/live/{session_id}/stream"})
    except Exception as exc:
        return JSONResponse({"error": f"Could not start simulated live monitoring: {exc}"}, status_code=500)


async def live_status(request: Request):
    session_id = request.path_params["session_id"]
    with live_lock:
        session = live_sessions.get(session_id)
        if not session:
            return JSONResponse({"error": "Live session not found."}, status_code=404)
        keys = ("state", "frame_number", "loop_count", "detection_count", "track_count", "error",
                "processed_frames", "inference_ms", "mean_inference_ms", "processing_fps", "elapsed_seconds",
                "unique_track_ids_this_loop", "source_fps", "frame_width", "frame_height", "hardware",
                "count_history", "centroid_history")
        return JSONResponse({key: session.get(key) for key in keys})


async def stop_live(request: Request):
    session_id = request.path_params["session_id"]
    with live_lock:
        session = live_sessions.get(session_id)
        if session:
            if session.get("state") in {"loading", "live"}:
                session["state"] = "stopping"
            session["stop"].set()
    return JSONResponse({"stopping": bool(session)})


async def live_stream(request: Request):
    session_id = request.path_params["session_id"]
    with live_lock:
        session = live_sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": "Live session not found."}, status_code=404)
    async def frames():
        last_frame = None
        while not session["stop"].is_set() and session.get("state") not in {"stopped", "error"}:
            with live_lock:
                frame = session.get("frame")
            if frame and frame != last_frame:
                last_frame = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.03)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


async def result_preview(request: Request):
    return result_file(request, "live_preview.jpg", "image/jpeg")


def result_file(request: Request, filename: str, media_type: str, download_name: str | None = None):
    path = RUNS_DIR / request.path_params["job_id"] / filename
    if not path.is_file():
        return JSONResponse({"error": "This result is not ready."}, status_code=404)
    return FileResponse(path, media_type=media_type, filename=download_name)


def convert_video_to_webm(source: Path, destination: Path):
    """Create a browser-playable VP8 copy for older MP4V outputs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("The annotated video could not be opened for browser conversion.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The annotated video has invalid dimensions.")
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("This OpenCV installation cannot create VP8 WebM video.")
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        writer.release()
    check = cv2.VideoCapture(str(destination))
    ok, _ = check.read()
    check.release()
    if not frames or not ok:
        destination.unlink(missing_ok=True)
        raise RuntimeError("The browser-compatible WebM could not be verified after conversion.")


async def result_video(request: Request):
    run_dir = RUNS_DIR / request.path_params["job_id"]
    webm_path = run_dir / "seal_annotated.webm"
    if webm_path.is_file():
        return result_file(request, "seal_annotated.webm", "video/webm", "seal_annotated.webm")
    mp4_path = run_dir / "seal_annotated.mp4"
    if not mp4_path.is_file():
        return JSONResponse({"error": "No annotated video is available for this run."}, status_code=404)
    try:
        run_metrics = json.loads((run_dir / "seal_benchmark.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        run_metrics = {}
    if run_metrics.get("video_encoder") == "NVIDIA NVENC H.264":
        return result_file(request, "seal_annotated.mp4", "video/mp4", "seal_annotated.mp4")
    try:
        with video_conversion_lock:
            if not webm_path.is_file():
                await run_in_threadpool(convert_video_to_webm, mp4_path, webm_path)
    except Exception as exc:
        return JSONResponse({"error": f"Could not prepare browser-playable video: {exc}"}, status_code=500)
    return result_file(request, "seal_annotated.webm", "video/webm", "seal_annotated.webm")


async def result_csv(request: Request):
    return result_file(request, "seal_tracks.csv", "text/csv", "seal_tracks.csv")


async def result_benchmark(request: Request):
    return result_file(request, "seal_benchmark.json", "application/json", "seal_benchmark.json")


def fetch_obis_occurrences(scientific_name: str, longitude: float, latitude: float, radius_km: float) -> dict:
    """Fetch bounded global points and a radius-filtered local query from OBIS."""
    cache_key = f"{scientific_name.casefold()}|{longitude:.4f}|{latitude:.4f}|{radius_km:g}"
    now = time.time()
    with occurrence_cache_lock:
        cached = occurrence_cache.get(cache_key)
        if cached and now - cached[0] < 900:
            return cached[1]
    headers = {"User-Agent": "Seal-OceanHackathon/1.0", "Accept": "application/json"}
    def query_obis(params):
        query = urlencode(params)
        request = UrlRequest(f"https://api.obis.org/v3/occurrence?{query}", headers=headers)
        with urlopen(request, timeout=12) as response:
            return json.loads(response.read(4_000_000).decode("utf-8"))
    global_data = query_obis({"scientificname": scientific_name, "size": 100})
    # Query the actual entered radius separately so a geographically clustered global sample
    # cannot be mistaken for evidence of local absence.
    dlat = radius_km / 110.574
    dlon = min(180.0, radius_km / (111.32 * max(0.1, abs(math.cos(math.radians(latitude))))))
    west, east = max(-180.0, longitude - dlon), min(180.0, longitude + dlon)
    south, north = max(-90.0, latitude - dlat), min(90.0, latitude + dlat)
    geometry = f"POLYGON (({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"
    local_data = query_obis({"scientificname": scientific_name, "geometry": geometry, "size": 40})
    def parse_points(payload, is_local=False):
        output = []
        results = payload.get("results", []) if isinstance(payload, dict) else []
        for row in results:
            try:
                lon, lat = float(row.get("decimalLongitude")), float(row.get("decimalLatitude"))
            except (TypeError, ValueError):
                continue
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                continue
            output.append({"lon": lon, "lat": lat, "date": str(row.get("eventDate") or "")[:10],
                           "scientific_name": row.get("scientificName") or scientific_name,
                           "dataset": row.get("datasetName") or "OBIS dataset", "is_local": is_local})
        return output
    global_points = parse_points(global_data)
    local_points = parse_points(local_data, True)
    for point in local_points:
        dlat_rad = math.radians(point["lat"] - latitude)
        dlon_rad = math.radians(point["lon"] - longitude)
        value = math.sin(dlat_rad / 2) ** 2 + math.cos(math.radians(latitude)) * math.cos(math.radians(point["lat"])) * math.sin(dlon_rad / 2) ** 2
        point["is_local"] = 6371.0088 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value)) <= radius_km
    points = []
    seen_points = set()
    for point in global_points + local_points:
        signature = (round(point["lon"], 5), round(point["lat"], 5), point["date"], point["dataset"])
        if signature in seen_points:
            continue
        seen_points.add(signature)
        points.append(point)
    result = {"total": int(global_data.get("total", 0)),
              "local_window_total": int(local_data.get("total", 0)), "points": points}
    with occurrence_cache_lock:
        occurrence_cache[cache_key] = (now, result)
    return result


async def species_range(request: Request):
    try:
        body = await request.json()
        job_id = str(body.get("job_id", ""))
        if not job_id or Path(job_id).name != job_id or not (RUNS_DIR / job_id / "seal_benchmark.json").is_file():
            return JSONResponse({"error": "The selected analysis run could not be found. Process a video before saving species location context."}, status_code=404)
        lon, lat = float(body.get("longitude")), float(body.get("latitude"))
        radius = float(body.get("radius_km", 100))
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            return JSONResponse({"error": "Longitude must be −180 to 180 and latitude −90 to 90."}, status_code=400)
        if radius not in {50, 100, 250, 500}:
            return JSONResponse({"error": "Choose a comparison radius of 50, 100, 250, or 500 km."}, status_code=400)
        names = body.get("species", [])
        if not isinstance(names, list):
            return JSONResponse({"error": "Species must be a list of scientific names."}, status_code=400)
        names = list(dict.fromkeys(str(name).strip()[:120] for name in names if str(name).strip()))[:12]
        if not names:
            return JSONResponse({"error": "This run has no labelled species candidates to compare."}, status_code=400)
        async def get_one(name):
            try:
                data = await run_in_threadpool(fetch_obis_occurrences, name, lon, lat, radius)
                return {"species": name, **data}
            except Exception as exc:
                return {"species": name, "error": f"OBIS lookup failed: {exc}", "total": 0, "points": []}
        datasets = await asyncio.gather(*(get_one(name) for name in names))
        payload = {"longitude": lon, "latitude": lat, "radius_km": radius,
                   "source": "OBIS API v3", "species": datasets,
                   "looked_up_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if not any(item.get("points") or item.get("local_window_total") for item in datasets):
            return JSONResponse({"error": "The occurrence lookup returned no usable data, so nothing was added to the report."}, status_code=502)
        (RUNS_DIR / job_id / "species_range.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return JSONResponse(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return JSONResponse({"error": "Enter valid camera longitude and latitude."}, status_code=400)


async def evaluate_job_tracking(request: Request):
    job_id = request.path_params["job_id"]
    workdir = RUNS_DIR / job_id
    metrics_path = workdir / "seal_benchmark.json"
    prediction_path = workdir / "seal_tracks.csv"
    if not metrics_path.is_file() or not prediction_path.is_file():
        return JSONResponse({"error": "Finish processing a video before evaluating its tracks."}, status_code=404)
    try:
        form = await request.form()
        ground_truth = form.get("ground_truth")
        if not isinstance(ground_truth, UploadFile) or not ground_truth.filename:
            return JSONResponse({"error": "Upload the labelled MOT ground-truth .txt file."}, status_code=400)
        if Path(ground_truth.filename).suffix.lower() not in {".txt", ".csv"}:
            return JSONResponse({"error": "Ground truth must be a MOTChallenge .txt or .csv file."}, status_code=400)
        raw = await ground_truth.read(50 * 1024 * 1024 + 1)
        if len(raw) > 50 * 1024 * 1024:
            return JSONResponse({"error": "Ground-truth file is larger than 50 MB."}, status_code=413)
        gt_text = raw.decode("utf-8-sig")
        prediction_text = prediction_path.read_text(encoding="utf-8-sig")
        result = await run_in_threadpool(evaluate_tracking, gt_text, prediction_text)
        (workdir / "ground_truth_mot.txt").write_text(gt_text, encoding="utf-8")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["tracking_validation"] = {**result, "ground_truth_filename": Path(ground_truth.filename).name}
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        bundle_path = workdir / "challenge_tracking_bundle.zip"
        if bundle_path.is_file():
            replacement = workdir / "challenge_tracking_bundle.updated.zip"
            with zipfile.ZipFile(bundle_path, "r") as source, zipfile.ZipFile(replacement, "w", zipfile.ZIP_DEFLATED) as updated:
                for item in source.infolist():
                    if item.filename not in {"seal_benchmark.json", "ground_truth_mot.txt"}:
                        updated.writestr(item, source.read(item.filename))
                updated.write(metrics_path, arcname="seal_benchmark.json")
                updated.write(workdir / "ground_truth_mot.txt", arcname="ground_truth_mot.txt")
            replacement.replace(bundle_path)
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["metrics"] = metrics
        return JSONResponse(metrics["tracking_validation"])
    except UnicodeDecodeError:
        return JSONResponse({"error": "Ground-truth file must be plain UTF-8 text."}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Could not evaluate tracking metrics: {exc}"}, status_code=500)


async def result_artifact(request: Request):
    artifacts = {
        "bundle.zip": ("challenge_tracking_bundle.zip", "application/zip"),
        "mot.txt": ("tracks_mot.txt", "text/plain"),
        "centroids.csv": ("relative_positions.csv", "text/csv"),
        "counts.csv": ("fish_count_per_frame.csv", "text/csv"),
        "heatmap.png": ("fish_presence_heatmap.png", "image/png"),
        "trajectories.png": ("trajectories_pixels.png", "image/png"),
    }
    name = request.path_params["artifact"]
    if name not in artifacts:
        return JSONResponse({"error": "Unknown result artifact."}, status_code=404)
    filename, media_type = artifacts[name]
    return result_file(request, filename, media_type, filename)


def build_report_pdf(title: str, subtitle: str, metric_rows: list[tuple[str, str]],
                     species_rows: list[tuple[str, str]] | None = None,
                     count_points: list[tuple[float, int]] | None = None,
                     heatmap_path: Path | None = None, trajectory_path: Path | None = None,
                     checklist_rows: list[tuple[str, str]] | None = None,
                     caveat: str = "", centroids: list[dict] | None = None,
                     species_range: dict | None = None, selected_range_species: str | None = None) -> bytes:
    """Render a compact, multi-page CHE01 report without slowing inference."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output = BytesIO()
    with PdfPages(output) as pdf:
        if species_range:
            range_species = species_range.get("species") or []
            selected = next((item for item in range_species if item.get("species") == selected_range_species), None)
            selected = selected or next((item for item in range_species if item.get("points")), None)
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f5fafb")
            fig.text(.07, .955, "REGIONAL SPECIES OCCURRENCE CONTEXT", color="#0b5873", fontsize=16, weight="bold")
            fig.text(.07, .928, "OBIS records near the submitted camera location and a bounded global point sample", color="#66808e", fontsize=9)
            fig.text(.07, .895, f"Camera: {species_range.get('latitude', '—')}° latitude, {species_range.get('longitude', '—')}° longitude  ·  Comparison radius: {species_range.get('radius_km', '—')} km",
                     color="#153b4d", fontsize=9, weight="bold")
            fig.text(.07, .872, f"Lookup time: {species_range.get('looked_up_at', '—')}  ·  Source: Ocean Biodiversity Information System (OBIS)",
                     color="#718895", fontsize=8)
            if selected:
                fig.text(.07, .825, f"Selected species: {selected.get('species', 'Unknown')}", color="#087fa7", fontsize=12, weight="bold")
                points = selected.get("points") or []
                map_ax = fig.add_axes([.09, .50, .82, .28])
                if points:
                    global_pts = [point for point in points if not point.get("is_local")]
                    local_pts = [point for point in points if point.get("is_local")]
                    if global_pts:
                        map_ax.scatter([p["lon"] for p in global_pts], [p["lat"] for p in global_pts], s=12, alpha=.48, color="#168bab", label="OBIS point sample")
                    if local_pts:
                        map_ax.scatter([p["lon"] for p in local_pts], [p["lat"] for p in local_pts], s=24, alpha=.8, color="#0b9a87", label="Within selected radius")
                    map_ax.scatter([species_range["longitude"]], [species_range["latitude"]], marker="*", s=130, color="#ef7f3c", edgecolor="white", linewidth=.6, label="Camera")
                    map_ax.set_xlim(-180, 180); map_ax.set_ylim(-90, 90)
                    map_ax.set_xticks([-180, -120, -60, 0, 60, 120, 180]); map_ax.set_yticks([-60, -30, 0, 30, 60])
                    map_ax.grid(alpha=.2); map_ax.legend(loc="lower left", fontsize=7, frameon=True)
                else:
                    map_ax.text(.5, .5, "No georeferenced coordinates returned", ha="center", va="center", transform=map_ax.transAxes)
                map_ax.set_title("Occurrence coordinates · sampled points", loc="left", color="#153b4d", fontsize=10)
                map_ax.set_xlabel("Longitude (degrees)"); map_ax.set_ylabel("Latitude (degrees)")
                month_counts = [0] * 12
                for point in points:
                    try:
                        month = int(str(point.get("date", ""))[5:7])
                        if 1 <= month <= 12:
                            month_counts[month - 1] += 1
                    except (TypeError, ValueError):
                        pass
                month_ax = fig.add_axes([.12, .34, .76, .10])
                month_ax.bar(range(1, 13), month_counts, color="#168bab")
                month_ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
                month_ax.set_ylabel("Records"); month_ax.grid(axis="y", alpha=.2)
                month_ax.set_title("Month in returned record sample (all years pooled)", loc="left", color="#153b4d", fontsize=9)
            table_rows = []
            for item in range_species[:12]:
                pts = item.get("points") or []
                local_sample = sum(1 for point in pts if point.get("is_local"))
                table_rows.append([item.get("species", "Unknown"), f"{int(item.get('total', 0)):,}", f"{int(item.get('local_window_total', 0)):,}", str(local_sample)])
            if table_rows:
                table_ax = fig.add_axes([.08, .105, .84, .19]); table_ax.axis("off")
                table = table_ax.table(cellText=table_rows, colLabels=["Candidate species", "OBIS global total", "Regional window", "Local sampled points"],
                                       colWidths=[.45, .18, .18, .19], cellLoc="left", loc="upper left")
                table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 1.35)
                for (row_index, _col), cell in table.get_celld().items():
                    cell.set_edgecolor("#d8e8ed")
                    if row_index == 0:
                        cell.set_facecolor("#0b5873"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
                    else:
                        cell.set_facecolor("#ffffff" if row_index % 2 else "#edf6f8"); cell.get_text().set_color("#153b4d")
            fig.text(.07, .065, "OBIS occurrence records show reported observations, not confirmed native ranges or migration routes. Global points are a bounded API sample; a missing local record does not prove absence. Verify candidates with regional experts before assigning non-native or invasive status.",
                     color="#526e7d", fontsize=7.4, wrap=True)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f5fafb")
        fig.text(.07, .93, "SEAL  /  CHE01 ADVANCED REPORT", color="#087fa7", fontsize=10, weight="bold")
        fig.text(.07, .885, title[:90], color="#0b5873", fontsize=21, weight="bold")
        fig.text(.07, .835, subtitle[:160], color="#557180", fontsize=10)
        fig.text(.07, .795, "RUN SUMMARY", color="#087fa7", fontsize=9, weight="bold")
        rows = metric_rows or [("Status", "No metrics available")]
        y = .765
        for index, (label, value) in enumerate(rows[:18]):
            col = index % 2
            if index and col == 0:
                y -= .067
            x = .07 + col * .46
            fig.text(x, y, str(label)[:48], color="#6b828d", fontsize=8)
            fig.text(x, y - .025, str(value)[:58], color="#153b4d", fontsize=11, weight="bold")
        fig.text(.07, .16, f"Species candidates in this run: {len(species_rows or [])}",
                 color="#153b4d", fontsize=10, weight="bold")
        if caveat:
            fig.text(.07, .07, caveat[:220], color="#718895", fontsize=7.5, wrap=True)
        fig.text(.07, .035, "Positions are image-pixel estimates. Model labels are predictions, not verified IDs.",
                 color="#718895", fontsize=7)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        if species_rows:
            for page_index in range(0, len(species_rows), 36):
                batch = species_rows[page_index:page_index + 36]
                fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f5fafb")
                fig.text(.07, .95, "SPECIES CANDIDATES BY TRACK", color="#0b5873", fontsize=17, weight="bold")
                fig.text(.07, .925, "Closed-set classifier predictions; unlabelled tracks are flagged for review.", color="#66808e", fontsize=9)
                table_ax = fig.add_axes([.07, .12, .86, .77]); table_ax.axis("off")
                table = table_ax.table(cellText=[[name, count] for name, count in batch],
                                       colLabels=["Candidate species / status", "Track count"],
                                       colWidths=[.76, .24], cellLoc="left", loc="upper left")
                table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.42)
                for (row_index, _col), cell in table.get_celld().items():
                    cell.set_edgecolor("#d8e8ed")
                    if row_index == 0:
                        cell.set_facecolor("#0b5873"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
                    else:
                        cell.set_facecolor("#ffffff" if row_index % 2 else "#edf6f8"); cell.get_text().set_color("#153b4d")
                fig.text(.07, .07, f"Batch {page_index // 36 + 1} · species categories {page_index + 1}–{page_index + len(batch)} of {len(species_rows)}", color="#718895", fontsize=8)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f5fafb")
        fig.text(.07, .96, "SPATIO-TEMPORAL OUTPUTS", color="#0b5873", fontsize=17, weight="bold")
        fig.text(.07, .935, "Frame counts, spatial distribution, track paths and CHE01 deliverable status", color="#66808e", fontsize=9)
        ax_count = fig.add_axes([.11, .69, .78, .19])
        if count_points:
            xs, ys = zip(*count_points)
            ax_count.plot(xs, ys, color="#0788ad", linewidth=1.6)
            ax_count.fill_between(xs, ys, color="#0788ad", alpha=.12)
        else:
            ax_count.text(.5, .5, "No frame-count series available", ha="center", va="center", transform=ax_count.transAxes)
        ax_count.set_title("Fish detections over video time", loc="left", color="#153b4d", fontsize=11)
        ax_count.set_xlabel("Time (seconds)"); ax_count.set_ylabel("Detections")
        ax_count.grid(alpha=.18)
        heat_ax = fig.add_axes([.08, .34, .40, .28])
        path_ax = fig.add_axes([.53, .34, .40, .28])
        if heatmap_path and heatmap_path.is_file():
            heat_ax.imshow(plt.imread(heatmap_path)); heat_ax.axis("off")
        elif centroids:
            heat_ax.hist2d([point.get("x", 0) for point in centroids],
                           [point.get("y", 0) for point in centroids], bins=18,
                           range=[[0, 1], [0, 1]], cmap="Blues")
            heat_ax.invert_yaxis(); heat_ax.set_xlim(0, 1); heat_ax.set_ylim(1, 0)
            heat_ax.set_xlabel("Normalized frame x"); heat_ax.set_ylabel("Normalized frame y")
        else:
            heat_ax.text(.5, .5, "Not available for this feed", ha="center", va="center", transform=heat_ax.transAxes)
            heat_ax.set_xticks([]); heat_ax.set_yticks([])
        if not (heatmap_path and heatmap_path.is_file()):
            heat_ax.set_title("Fish presence heatmap", loc="left", color="#153b4d", fontsize=10)
        if trajectory_path and trajectory_path.is_file():
            path_ax.imshow(plt.imread(trajectory_path)); path_ax.axis("off")
        elif centroids:
            groups = defaultdict(list)
            for point in centroids:
                if point.get("track_id") is not None:
                    groups[(point.get("loop", 1), point["track_id"])].append(point)
            for points in list(groups.values())[-24:]:
                if len(points) > 1:
                    path_ax.plot([point.get("x", 0) for point in points],
                                 [point.get("y", 0) for point in points], linewidth=1)
            path_ax.invert_yaxis(); path_ax.set_xlim(0, 1); path_ax.set_ylim(1, 0)
            path_ax.set_xlabel("Normalized frame x"); path_ax.set_ylabel("Normalized frame y")
        else:
            path_ax.text(.5, .5, "Not available for this feed", ha="center", va="center", transform=path_ax.transAxes)
            path_ax.set_xticks([]); path_ax.set_yticks([])
        if not (trajectory_path and trajectory_path.is_file()):
            path_ax.set_title("Track trajectories", loc="left", color="#153b4d", fontsize=10)
        fig.text(.08, .29, "CHE01 DELIVERABLE STATUS", color="#087fa7", fontsize=9, weight="bold")
        for index, (label, status) in enumerate((checklist_rows or [])[:8]):
            y = .265 - index * .022
            fig.text(.08, y, str(label)[:74], color="#254a5b", fontsize=7.5)
            fig.text(.91, y, str(status)[:30], color="#087fa7", fontsize=7.5, ha="right")
        fig.text(.08, .065, "Accuracy / tracking scores need labelled validation data. Background detections are a false-positive check, not precision.", color="#526e7d", fontsize=7.5, wrap=True)
        fig.text(.08, .035, "Generated by the local Seal prototype · CHE01 deliverable-oriented summary", color="#718895", fontsize=7.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
    return output.getvalue()


def make_job_report(workdir: Path, selected_range_species: str | None = None) -> bytes:
    metrics = json.loads((workdir / "seal_benchmark.json").read_text(encoding="utf-8"))
    species = metrics.get("species_classification") or {}
    bg = metrics.get("background_video_check") or {}
    tracking = metrics.get("tracking_validation") or {}
    tracking_summary = (f"HOTA {tracking['hota_percent']:.2f}% · IDF1 {tracking['idf1_percent']:.2f}% · MOTA {tracking['mota_percent']:.2f}%"
                        if tracking.get("status") == "scored" else "Requires ground-truth tracks")
    metrics_rows = [
        ("Input video", metrics.get("source_filename", "Uploaded video")),
        ("Model", metrics.get("model_file", "Fishial detector")),
        ("Resolution / source rate", f"{metrics.get('resolution', '—')} / {metrics.get('video_fps', '—')} FPS"),
        ("Frames analyzed", f"{metrics.get('frames_processed', 0):,}"),
        ("Processing speed", f"{metrics.get('processing_fps', '—')} FPS"),
        ("Mean inference", f"{metrics.get('mean_inference_ms_per_frame', '—')} ms / frame"),
        ("Total processing time", f"{metrics.get('total_elapsed_seconds', metrics.get('elapsed_seconds', '—'))} s"),
        ("Hardware", metrics.get("hardware", "Not reported")),
        ("Detection rows", f"{metrics.get('detection_rows', 0):,}"),
        ("Mean detections / frame", metrics.get("mean_detections_per_frame", "—")),
        ("Unique track IDs", metrics.get("unique_track_ids", "—")),
        ("Mean detector confidence", metrics.get("mean_confidence", "—")),
        ("Confidence threshold", metrics.get("confidence_threshold", "—")),
        ("Enhancement / image size", f"{metrics.get('frame_enhancement', 'original')} / {metrics.get('inference_image_size', '—')}"),
        ("Precision / Recall / mAP", "Requires labelled validation set"),
        ("HOTA / IDF1 / MOTA", tracking_summary),
        ("Background detections", f"{bg.get('detection_count', 'Not run')}"),
    ]
    if tracking.get("status") == "scored":
        metrics_rows.append(("ID switches / FP / FN", f"{tracking['identity_switches']} / {tracking['false_positives_iou_0_5']} / {tracking['false_negatives_iou_0_5']}"))
    decisions = species.get("species_counts") or []
    species_rows = []
    for item in decisions:
        count = int(item.get("track_count", 0))
        species_rows.append((item.get("species", "Unknown"), f"{count} track" if count == 1 else f"{count} tracks"))
    checklist = [
        ("MOT tracking results ZIP", "Included" if (workdir / "challenge_tracking_bundle.zip").exists() else "Unavailable"),
        ("Annotated output video", "Included" if any((workdir / name).is_file() for name in ("seal_annotated.mp4", "seal_annotated.webm")) else "Unavailable"),
        ("Trajectory, heatmap, per-frame counts, relative positions", "Included in ZIP" if (workdir / "challenge_tracking_bundle.zip").exists() else "Unavailable"),
        ("Detection accuracy metrics", "Pending labelled validation"),
        ("Fish-free background video check", "Included" if bg else "Not supplied"),
        ("Source code and README", "See project repository / README"),
        ("5-minute pitch deck", "See project pitch folder"),
        ("Dataset citation in pitch", "Add dataset/source citation"),
    ]
    rows = []
    csv_path = workdir / "seal_tracks.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    fps = float(metrics.get("video_fps") or 25.0)
    frame_count = max(1, int(metrics.get("frames_processed") or 1))
    bin_size = max(1, round(fps * 2))
    bins = [0] * max(1, (frame_count + bin_size - 1) // bin_size)
    for row in rows:
        try:
            bins[min(len(bins) - 1, max(0, (int(float(row.get("frame", 1))) - 1) // bin_size))] += 1
        except (TypeError, ValueError):
            pass
    count_points = [(index * bin_size / fps, count) for index, count in enumerate(bins)]
    range_path = workdir / "species_range.json"
    range_data = json.loads(range_path.read_text(encoding="utf-8")) if range_path.is_file() else None
    if range_data and selected_range_species not in {item.get("species") for item in range_data.get("species", [])}:
        selected_range_species = None
    return build_report_pdf("Video analysis report", metrics.get("source_filename", "Uploaded underwater video"),
                            metrics_rows, species_rows, count_points,
                            workdir / "fish_presence_heatmap.png", workdir / "trajectories_pixels.png",
                            checklist, metrics.get("note", ""),
                            species_range=range_data, selected_range_species=selected_range_species)


async def job_report(request: Request):
    workdir = RUNS_DIR / request.path_params["job_id"]
    report_path = workdir / "seal_report.pdf"
    if not (workdir / "seal_benchmark.json").is_file():
        return JSONResponse({"error": "The report is available after detection and species labelling finish."}, status_code=404)
    try:
        content = await run_in_threadpool(make_job_report, workdir, request.query_params.get("range_species"))
        report_path.write_bytes(content)
        return FileResponse(report_path, media_type="application/pdf", filename="seal_analysis_report.pdf")
    except Exception as exc:
        return JSONResponse({"error": f"Could not create the report PDF: {exc}"}, status_code=500)


async def live_report(request: Request):
    session_id = request.path_params["session_id"]
    with live_lock:
        session = live_sessions.get(session_id)
    folder = RUNS_DIR / "live" / session_id
    if session is None:
        stats_path = folder / "live_stats.json"
        if not stats_path.is_file():
            return JSONResponse({"error": "Live session not found. Start a feed before exporting its report."}, status_code=404)
        snapshot = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        snapshot = dict(session)
        snapshot["count_history"] = list(session.get("count_history", []))
        snapshot["centroid_history"] = list(session.get("centroid_history", []))
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / "seal_live_report.pdf"
    try:
        points = [(item.get("t", 0), item.get("detections", 0)) for item in snapshot["count_history"]]
        metrics_rows = [
                ("Feed type", "Simulated looped clip"), ("Input video", Path(snapshot.get("video_path", "clip")).name),
                ("Sampled frames processed", f"{snapshot.get('processed_frames', 0):,}"),
                ("Mean inference", f"{snapshot.get('mean_inference_ms', '—')} ms"),
                ("Processing rate", f"{snapshot.get('processing_fps', '—')} FPS"),
                ("Feed elapsed", f"{snapshot.get('elapsed_seconds', '—')} s"),
                ("Hardware", snapshot.get("hardware", "Not reported")),
                ("Current detections / tracks", f"{snapshot.get('detection_count', 0)} / {snapshot.get('track_count', 0)}"),
                ("Clip loop", str(snapshot.get("loop_count", 0))),
                ("Precision / Recall / mAP", "Requires labelled validation set"),
                ("HOTA / IDF1 / MOTA", "Requires ground-truth tracks"),
        ]
        content = await run_in_threadpool(build_report_pdf, "Live monitoring report", "Simulated live feed · not a connected underwater camera",
                                           metrics_rows, [], points, None, None,
                                           [("Live annotated stream", "Simulated clip"),
                                            ("Runtime and spatial samples", "Included in this report"),
                                            ("Accuracy scores", "Pending labelled validation")],
                                           "Track IDs are temporary ByteTrack assignments. Positions are image-pixel estimates.",
                                           snapshot.get("centroid_history", []))
        report_path.write_bytes(content)
        return FileResponse(report_path, media_type="application/pdf", filename="seal_live_report.pdf")
    except Exception as exc:
        return JSONResponse({"error": f"Could not create the live report PDF: {exc}"}, status_code=500)


async def health(request: Request):
    return JSONResponse({"ok": True, "default_model_available": DEFAULT_WEIGHTS.exists()})


routes = [
    Route("/api/health", health),
    Route("/api/live/demo", live_demo_info),
    Route("/api/live/active", active_live),
    Route("/api/live/start", start_live, methods=["POST"]),
    Route("/api/live/{session_id}/report.pdf", live_report),
    Route("/api/live/{session_id}/stream", live_stream),
    Route("/api/live/{session_id}/stop", stop_live, methods=["POST"]),
    Route("/api/live/{session_id}", live_status),
    Route("/api/jobs", create_job, methods=["POST"]),
    Route("/api/jobs/{job_id}", job_status),
    Route("/api/latest", latest_job),
    Route("/api/jobs/{job_id}/preview.jpg", result_preview),
    Route("/api/jobs/{job_id}/video", result_video),
    Route("/api/jobs/{job_id}/tracks.csv", result_csv),
    Route("/api/jobs/{job_id}/benchmark.json", result_benchmark),
    Route("/api/species/range", species_range, methods=["POST"]),
    Route("/api/jobs/{job_id}/tracking-metrics", evaluate_job_tracking, methods=["POST"]),
    Route("/api/jobs/{job_id}/report.pdf", job_report),
    Route("/api/jobs/{job_id}/{artifact}", result_artifact),
    Mount("/", app=StaticFiles(directory=WEB_DIR, html=True), name="web"),
]
app = Starlette(debug=False, routes=routes)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("SEAL_PORT", "8502")), log_level="info")
