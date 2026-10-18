"""Shared low-level inference helpers for the local FishTrack web runner."""
import csv
import json
import platform
import time
import zipfile
import zlib
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from video_encoding import create_video_writer


_TRACK_COLORS = [(45, 185, 235), (220, 145, 45), (82, 190, 105), (205, 90, 180),
                 (45, 145, 235), (185, 185, 55), (190, 105, 70), (80, 190, 195)]


def track_color(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color for a ByteTrack ID."""
    return _TRACK_COLORS[int(track_id) % len(_TRACK_COLORS)]


def draw_track_trails(frame, trails, frame_idx=None, last_seen=None, max_gap=6):
    """Draw translucent recent centroid paths; coordinates are image pixels."""
    overlay = frame.copy()
    for track_id, points in trails.items():
        if not points:
            continue
        if frame_idx is not None and last_seen is not None and frame_idx - last_seen.get(track_id, frame_idx) > max_gap:
            continue
        color = track_color(track_id)
        if len(points) > 1:
            polyline = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [polyline], False, color, 3, cv2.LINE_AA)
        cv2.circle(overlay, points[-1], 5, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)


def extract_model(archive_path: Path, destination: Path) -> Path:
    max_model_bytes = 2 * 1024**3
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            item for item in archive.infolist()
            if not item.is_dir() and PurePosixPath(item.filename).suffix.lower() in {".pt", ".torchscript"}
        ]
        if not members:
            raise ValueError("This ZIP has no .pt or .torchscript detector. Upload detector weights, not a dataset or classifier archive.")
        members.sort(key=lambda item: ("detector" not in item.filename.lower(), -item.file_size))
        member = members[0]
        relative = PurePosixPath(member.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("The model archive contains an unsafe file path.")
        if member.file_size <= 0 or member.file_size > max_model_bytes:
            raise ValueError("The model file in this ZIP is empty or larger than the 2 GB limit.")
        output_path = destination / f"fishial_detector{relative.suffix.lower()}"
        with archive.open(member) as source, output_path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        return output_path


def resolve_model_path(upload_path: Path, destination: Path) -> Path:
    if upload_path.suffix.lower() != ".zip":
        return upload_path
    try:
        return extract_model(upload_path, destination)
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded ZIP is invalid or incomplete.") from exc


def _prepare_inference_frame(frame, enhancement: str):
    if enhancement == "original":
        return frame
    if enhancement != "clahe":
        raise ValueError("Unsupported frame enhancement. Choose original or clahe.")
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def run_tracking(model, video_path: Path, output_path: Path, confidence: float, progress_callback=None,
                 enhancement: str = "original", imgsz: int = 640, preview_callback=None,
                 render_output: bool = True):
    """Run the same frame-wise YOLO tracking used in FishTrack, without UI dependencies."""
    # Reuse detector weights across jobs, but do not carry tracker history between videos.
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        if hasattr(tracker, "reset"):
            tracker.reset()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open this video. Try exporting it as MP4 (H.264) and upload again.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("The uploaded video has no readable frames.")
    writer = None
    if render_output:
        writer = create_video_writer(output_path, fps, width, height)
        if not writer.isOpened():
            cap.release()
            raise ValueError("Could not create the annotated video.")

    fields = ["frame", "time_seconds", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
    rows, tracked_ids = [], set()
    trails = defaultdict(lambda: deque(maxlen=60))
    trail_last_seen = {}
    reference_frame = None
    frame_idx = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if reference_frame is None and render_output:
                reference_frame = frame.copy()
            inference_frame = _prepare_inference_frame(frame, enhancement)
            try:
                result = model.track(inference_frame, persist=True, tracker="bytetrack.yaml", conf=confidence,
                                     imgsz=imgsz, verbose=False)[0]
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}".strip()
                raise RuntimeError(
                    "The checkpoint loaded, but Ultralytics could not run detection/tracking. "
                    f"Underlying error: {detail[:500]}"
                ) from exc
            inference_seconds += float((result.speed or {}).get("inference", 0.0)) / 1000.0
            boxes = result.boxes
            frame_detections = []
            if boxes is not None and len(boxes):
                xyxy, scores = boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for box, score, class_id, track_id in zip(xyxy, scores, class_ids, track_ids):
                    x1, y1, x2, y2 = box
                    names = model.names
                    class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    if track_id is not None:
                        track_id = int(track_id)
                    frame_detections.append((box, score, class_name, track_id))
                    rows.append({"frame": frame_idx, "time_seconds": round(frame_idx / fps, 3), "track_id": track_id,
                        "class_id": class_id, "class_name": class_name, "confidence": round(score, 4),
                        "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                        "center_x": round((x1 + x2) / 2, 2), "center_y": round((y1 + y2) / 2, 2)})
                    if track_id is not None:
                        tracked_ids.add(track_id)
                        trails[track_id].append((round((x1 + x2) / 2), round((y1 + y2) / 2)))
                        trail_last_seen[track_id] = frame_idx
            frame_idx += 1
            should_preview = preview_callback and (frame_idx % max(1, int(fps / 3)) == 0 or frame_idx == 1)
            if render_output or should_preview:
                # Draw only sampled previews when the species pass will produce the final annotated video.
                annotated = draw_track_trails(frame.copy(), trails, frame_idx - 1, trail_last_seen)
                for box, score, class_name, track_id in frame_detections:
                    x1, y1, x2, y2 = (int(round(value)) for value in box)
                    color = track_color(track_id) if track_id is not None else (0, 220, 180)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    label = f"ID {track_id} | {class_name} {score:.2f}" if track_id is not None else f"{class_name} {score:.2f}"
                    cv2.putText(annotated, label[:70], (max(0, x1), max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            if render_output:
                writer.write(annotated)
            if should_preview:
                preview_callback(annotated, frame_idx)
            if progress_callback and (frame_idx % max(1, int(fps)) == 0 or (total_frames and frame_idx == total_frames)):
                progress_callback(frame_idx, total_frames)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not frame_idx:
        raise ValueError("No frames could be decoded from this video.")
    if reference_frame is not None:
        cv2.imwrite(str(output_path.with_name("reference_frame.jpg")), reference_frame)
    elapsed = time.perf_counter() - started
    try:
        import torch
        hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or platform.machine()
        if torch.cuda.is_available():
            hardware = f"{hardware} (CUDA)"
        else:
            hardware = f"CPU · {hardware}"
    except Exception:
        hardware = f"{platform.system()} · {platform.machine()}"
    metrics = {"frames_processed": frame_idx, "video_fps": round(fps, 3), "resolution": f"{width}x{height}",
        "frame_width": width, "frame_height": height,
        "elapsed_seconds": round(elapsed, 3), "processing_fps": round(frame_idx / elapsed, 3) if elapsed else 0,
        "model_inference_seconds": round(inference_seconds, 3),
        "mean_inference_ms_per_frame": round(1000 * inference_seconds / frame_idx, 2),
        "detection_rows": len(rows), "mean_detections_per_frame": round(len(rows) / frame_idx, 3),
        "unique_track_ids": len(tracked_ids),
        "mean_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 4) if rows else None,
        "confidence_threshold": confidence, "frame_enhancement": enhancement, "inference_image_size": imgsz, "hardware": hardware,
        "note": "Runtime and output statistics only; accuracy requires ground-truth annotations for this video."}
    if writer is not None:
        metrics["video_encoder"] = writer.encoder
    return rows, fields, metrics


def save_csv(rows, fields, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_deliverable_bundle(rows, metrics: dict, output_dir: Path) -> dict[str, Path]:
    """Write challenge-ready tracking/spatial exports and a ZIP for one sequence."""
    width, height = int(metrics["frame_width"]), int(metrics["frame_height"])
    frame_count = int(metrics["frames_processed"])
    mot_path = output_dir / "tracks_mot.txt"
    centroid_path = output_dir / "relative_positions.csv"
    count_path = output_dir / "fish_count_per_frame.csv"
    heatmap_path = output_dir / "fish_presence_heatmap.png"
    trajectory_path = output_dir / "trajectories_pixels.png"

    counts = [0] * frame_count
    import numpy as np
    heatmap = np.zeros((height, width), dtype="uint32")
    trajectories: dict[int, list[tuple[int, int]]] = {}
    with mot_path.open("w", encoding="utf-8", newline="") as mot, centroid_path.open("w", encoding="utf-8", newline="") as centroids:
        centroid_writer = csv.writer(centroids)
        centroid_writer.writerow(["frame", "time_seconds", "track_id", "centroid_x_norm", "centroid_y_norm", "centroid_x_px", "centroid_y_px"])
        for row in rows:
            frame_zero = int(row["frame"])
            if 0 <= frame_zero < frame_count:
                counts[frame_zero] += 1
            track_id = row.get("track_id")
            x1, y1 = float(row["x1"]), float(row["y1"])
            x2, y2 = float(row["x2"]), float(row["y2"])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ix, iy = min(width - 1, max(0, round(cx))), min(height - 1, max(0, round(cy)))
            heatmap[iy, ix] += 1
            if track_id is None:
                continue
            track_id = int(track_id)
            # MOTChallenge convention: 1-based frame, x/y/width/height, confidence, then placeholders.
            mot.write(f"{frame_zero + 1},{track_id},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{float(row['confidence']):.4f},-1,-1,-1\n")
            centroid_writer.writerow([frame_zero + 1, row["time_seconds"], track_id, f"{cx / width:.6f}", f"{cy / height:.6f}", f"{cx:.2f}", f"{cy:.2f}"])
            trajectories.setdefault(track_id, []).append((ix, iy))

    with count_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_seconds", "fish_detection_count"])
        fps = float(metrics["video_fps"] or 25.0)
        writer.writerows((i + 1, f"{i / fps:.3f}", count) for i, count in enumerate(counts))

    reference_path = output_dir / "reference_frame.jpg"
    reference = cv2.imread(str(reference_path)) if reference_path.exists() else None
    if reference is not None and (reference.shape[1] != width or reference.shape[0] != height):
        reference = cv2.resize(reference, (width, height))
    base = reference.copy() if reference is not None else np.zeros((height, width, 3), dtype=np.uint8)
    if reference is None:
        base[:] = (25, 40, 48)
    cv2.putText(base, "Fish presence heatmap · image pixels", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 245, 255), 2, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(heatmap.astype("float32"), (0, 0), sigmaX=max(2, min(width, height) / 90))
    if blurred.max() > 0:
        scaled = np.uint8(np.clip(blurred / blurred.max() * 255, 0, 255))
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        base = cv2.addWeighted(base, 0.62, colored, 0.48, 0)
        cv2.putText(base, "Detection-centroid concentration (not calibrated distance)", (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(base, "No fish detections", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(heatmap_path), base)

    plot = reference.copy() if reference is not None else np.full((height, width, 3), (248, 250, 250), dtype=np.uint8)
    cv2.putText(plot, "Fish trajectories · image-pixel coordinates", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 50, 60), 2, cv2.LINE_AA)
    for track_id, points in trajectories.items():
        color = (64 + zlib.crc32(str(track_id).encode()) % 170, 60 + zlib.crc32(f"b{track_id}".encode()) % 170, 60 + zlib.crc32(f"c{track_id}".encode()) % 170)
        if len(points) > 1:
            cv2.polylines(plot, [np.asarray(points, dtype=np.int32)], False, color, max(2, min(width, height) // 500), cv2.LINE_AA)
        if points:
            cv2.circle(plot, points[0], max(3, min(width, height) // 140), color, -1)
            cv2.putText(plot, f"ID {track_id}", points[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(trajectory_path), plot)

    bundle_path = output_dir / "challenge_tracking_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (mot_path, centroid_path, count_path, heatmap_path, trajectory_path,
                     output_dir / "seal_annotated.webm", output_dir / "seal_annotated.mp4", output_dir / "seal_benchmark.json"):
            if path.exists():
                archive.write(path, arcname=path.name)
    return {"mot": mot_path, "centroids": centroid_path, "counts": count_path,
            "heatmap": heatmap_path, "trajectories": trajectory_path, "bundle": bundle_path}


def run_background_check(model, video_path: Path, confidence: float, enhancement: str = "original", imgsz: int = 640) -> dict:
    """Count detector outputs on a fish-free clip; this is a background false-positive check, not precision."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open the fish-free background clip.")
    frames, detections = 0, 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(_prepare_inference_frame(frame, enhancement), conf=confidence, imgsz=imgsz, verbose=False)[0]
            detections += len(result.boxes) if result.boxes is not None else 0
            frames += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("The fish-free background clip has no readable frames.")
    return {"frames_processed": frames, "detection_count": detections,
            "detections_per_frame": round(detections / frames, 4),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "confidence_threshold": confidence, "frame_enhancement": enhancement, "inference_image_size": imgsz,
            "note": "A detector output count on an assumed fish-free clip; not a precision score."}
