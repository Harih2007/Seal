import csv
import io
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

# Keep library settings local to the project so the app works in restricted
# environments that cannot write to the user's home directory.
APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("STREAMLIT_CONFIG_DIR", str(APP_DIR / ".streamlit"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(APP_DIR / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import altair as alt
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO


st.set_page_config(page_title="Seal · Underwater Vision", page_icon="🌊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#123447; --muted:#587181; --blue:#087fa7; --blue-dark:#075477; --aqua:#bdebf0; --line:#dcebf0; }
.stApp { background:#fbfdfe; color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(251,253,254,.92); }
[data-testid="stAppViewContainer"] .main .block-container { max-width:1180px; padding-top:1.1rem; padding-bottom:4rem; }
.brandbar { display:flex; align-items:center; justify-content:space-between; padding:8px 2px 18px; border-bottom:1px solid var(--line); margin-bottom:18px; }
.brand { font:800 25px Manrope,sans-serif; letter-spacing:-1.2px; color:#123447; }
.brand span { color:#0785a8; }
.brandnote { color:#66808e; font-size:12px; letter-spacing:.12em; text-transform:uppercase; }
[data-testid="stRadio"] > div { gap:.45rem; }
[data-testid="stRadio"] label { border:1px solid transparent; border-radius:999px; padding:7px 16px; color:#496777; font-weight:600; }
[data-testid="stRadio"] label:has(input:checked) { background:#e9f7fa; border-color:#b5e3eb; color:#076888; }
[data-testid="stRadio"] label p { font-size:14px; }
[data-testid="stRadio"] label > div:first-child, div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display:none !important; }
.hero { position:relative; overflow:hidden; min-height:315px; padding:40px 62px 64px; border:1px solid #e3eef2; border-radius:28px; background:linear-gradient(128deg,#fff 4%,#f1fbfd 68%,#e9f8fb 100%); box-shadow:0 14px 38px rgba(18,70,92,.06); }
.eyebrow { color:#087fa7; text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:800; }
.hero h1 { font:800 clamp(44px,6vw,76px)/1.02 Manrope,sans-serif; letter-spacing:-.055em; color:#103347; max-width:750px; margin:18px 0; }
.hero p { color:#526f7f; font-size:18px; line-height:1.65; max-width:650px; margin:0; }
.hero .seal { color:#0785a8; }
.wave { position:absolute; left:0; bottom:-2px; width:100%; height:88px; }
.section-title { font:800 30px Manrope,sans-serif; letter-spacing:-.035em; color:#123447; margin:0 0 8px; }
.section-copy { color:#607a88; font-size:15px; margin-bottom:24px; }
.info-card { border:1px solid #dcebf0; background:white; border-radius:18px; padding:22px 24px; height:100%; box-shadow:0 7px 24px rgba(18,70,92,.04); }
.info-card h3 { font:700 17px Manrope,sans-serif; color:#123447; margin:0 0 8px; }
.info-card p { color:#617b89; line-height:1.55; margin:0; font-size:14px; }
.tag { display:inline-block; border-radius:999px; padding:6px 11px; background:#eaf7fa; color:#087493; font-size:12px; font-weight:700; margin:0 5px 7px 0; }
div.stButton > button[kind="primary"] { border:0; border-radius:999px; background:#087fa7; color:white; padding:.72rem 1.4rem; font-weight:700; box-shadow:0 7px 18px rgba(8,127,167,.2); }
div.stButton > button[kind="primary"]:hover { background:#075477; color:white; border:0; }
div.stButton > button { border-radius:12px; }
.stFileUploader section { border:1.5px dashed #a9cbd6; border-radius:16px; background:#f7fcfd; }
.stFileUploader button { background:#087fa7; color:white; border:0; border-radius:9px; }
[data-testid="stMetric"] { background:white; border:1px solid #deedf1; border-radius:15px; padding:15px; }
</style>
<div class="brandbar"><div class="brand">Seal<span>.</span></div><div class="brandnote">Underwater vision · image-pixel tracking</div></div>
""", unsafe_allow_html=True)

PAGES = ["Home", "Upload", "Results", "About"]
if "page" not in st.session_state:
    st.session_state.page = st.query_params.get("page", "Home")
if st.session_state.page not in PAGES:
    st.session_state.page = "Home"

def go_to(page: str):
    st.session_state.page = page
    st.query_params["page"] = page

def remember_upload(widget_key: str, data_key: str, name_key: str):
    uploaded = st.session_state.get(widget_key)
    if uploaded is not None:
        st.session_state[data_key] = uploaded.getvalue()
        st.session_state[name_key] = uploaded.name

st.radio("Main menu", PAGES, horizontal=True, label_visibility="collapsed", key="page", on_change=lambda: st.query_params.update({"page": st.session_state.page}))
page = st.session_state.page

if page == "Home":
    st.markdown("""
    <section class="hero">
      <div class="eyebrow">See more beneath the surface</div>
      <h1>Meet <span class="seal">Seal</span>.</h1>
      <p>Explore underwater footage with AI-assisted fish detection and visual tracking. Bring a model and a clip, then review annotated video and image-pixel trajectories.</p>
      <svg class="wave" viewBox="0 0 1440 150" preserveAspectRatio="none" aria-hidden="true">
        <path d="M0,79 C170,139 270,31 448,69 C623,106 690,130 864,77 C1050,20 1171,119 1440,48 L1440,150 L0,150 Z" fill="#cceff3"/>
        <path d="M0,106 C172,56 285,146 464,94 C653,39 758,59 924,105 C1107,155 1255,48 1440,84 L1440,150 L0,150 Z" fill="#087fa7" opacity=".88"/>
        <path d="M0,127 C213,103 292,143 483,119 C686,94 814,130 1014,119 C1195,108 1283,96 1440,119 L1440,150 L0,150 Z" fill="#075477" opacity=".96"/>
      </svg>
    </section>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Upload a video and model", type="primary", on_click=go_to, args=("Upload",)):
        pass
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="info-card"><h3>Detect</h3><p>Run an Ultralytics fish detector on underwater frames and inspect its confidence scores.</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="info-card"><h3>Track</h3><p>Follow detections across frames with ByteTrack and temporary track IDs.</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="info-card"><h3>Review</h3><p>Watch annotated output, compare runtime, and inspect pixel-coordinate paths.</p></div>', unsafe_allow_html=True)
    st.caption("Prototype for exploration. Detection performance must be checked against labelled underwater footage.")

elif page == "Upload":
    st.markdown('<div class="section-title">Upload a model and a clip</div><div class="section-copy">Run one detector on a video, then inspect the annotated result and runtime summary.</div>', unsafe_allow_html=True)
    weights_upload = st.file_uploader("Model weights (.pt or ZIP archive)", type=["pt", "zip"], key="weights_widget", on_change=remember_upload, args=("weights_widget", "weights_bytes", "weights_name"))
    video_upload = st.file_uploader("Underwater video", type=["mp4", "mov", "avi", "mkv"], key="video_widget", on_change=remember_upload, args=("video_widget", "video_bytes", "video_name"))
    confidence = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)
    clahe_enabled = st.checkbox("Apply mild CLAHE contrast enhancement before detection", value=False)
    clahe_clip_limit = st.slider("CLAHE strength", 1.0, 3.0, 1.8, 0.2, disabled=not clahe_enabled)
    st.caption("CLAHE adjusts local contrast; it does not increase resolution. Output retains the original frame colors.")
    st.caption("Upload an Ultralytics YOLO detection checkpoint (.pt), or a ZIP containing one. Accuracy metrics require ground-truth annotations for this video.")


def extract_model(archive_path: Path, destination: Path) -> Path:
    """Extract one model file safely; reject suspicious or oversized ZIPs."""
    max_model_bytes = 2 * 1024**3
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            item for item in archive.infolist()
            if not item.is_dir() and PurePosixPath(item.filename).suffix.lower() in {".pt", ".torchscript"}
        ]
        if not members:
            raise ValueError("This ZIP has no .pt or .torchscript model. Check that you uploaded the Fishial detector archive, not a dataset or source-code ZIP.")
        members.sort(key=lambda item: ("detector" not in item.filename.lower(), -item.file_size))
        member = members[0]
        relative = PurePosixPath(member.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("The model archive contains an unsafe file path.")
        if member.file_size <= 0 or member.file_size > max_model_bytes:
            raise ValueError("The model file in this ZIP is empty or larger than the 2 GB safety limit.")

        # Write just the selected model file instead of extracting the whole archive.
        suffix = relative.suffix.lower()
        output_path = destination / f"fishial_detector{suffix}"
        with archive.open(member) as source, output_path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        return output_path


def get_model_path(upload_path: Path, destination: Path) -> Path:
    if upload_path.suffix.lower() == ".zip":
        try:
            return extract_model(upload_path, destination)
        except zipfile.BadZipFile as exc:
            raise ValueError("The uploaded ZIP is invalid or incomplete.") from exc
    return upload_path


def process_video(model: YOLO, video_path: Path, output_path: Path, confidence_threshold: float, clahe_enabled: bool = False, clahe_clip_limit: float = 1.8):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video. Try exporting it as MP4 (H.264) and uploading again.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("The uploaded video has no readable frames.")

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the annotated MP4 on this machine. Check that OpenCV video codecs are available.")

    fields = [
        "frame", "time_seconds", "track_id", "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y",
    ]
    rows = []
    frame_idx = 0
    inference_seconds = 0.0
    tracked_ids = set()
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8)) if clahe_enabled else None
    run_started = time.perf_counter()
    progress = st.progress(0.0)
    status = st.empty()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            inference_frame = frame
            if clahe is not None:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                inference_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            try:
                result = model.track(
                    inference_frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence_threshold,
                    verbose=False,
                )[0]
            except Exception as exc:
                raise RuntimeError(
                    "The checkpoint loaded, but Ultralytics could not run detection/tracking. "
                    "Confirm the checkpoint is an Ultralytics detection model and that ByteTrack dependencies are installed. "
                    "TorchScript exports may also be incompatible with some checkpoint operations."
                ) from exc

            writer.write(result.plot(img=frame.copy()))
            inference_seconds += float((result.speed or {}).get("inference", 0.0)) / 1000.0
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for box, score, class_id, track_id in zip(xyxy, scores, class_ids, track_ids):
                    x1, y1, x2, y2 = box
                    names = model.names
                    class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    rows.append({
                        "frame": frame_idx,
                        "time_seconds": round(frame_idx / fps, 3),
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(score, 4),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "center_x": round((x1 + x2) / 2, 2),
                        "center_y": round((y1 + y2) / 2, 2),
                    })
                    if track_id is not None:
                        tracked_ids.add(int(track_id))

            frame_idx += 1
            refresh_every = max(1, int(fps))
            if frame_idx % refresh_every == 0 or (total_frames and frame_idx == total_frames):
                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0))
                status.write(f"Processed {frame_idx} of {total_frames or '?'} frames…")
    finally:
        cap.release()
        writer.release()

    if frame_idx == 0:
        raise ValueError("No frames could be decoded from this video.")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError("The annotated video could not be written.")
    elapsed_seconds = time.perf_counter() - run_started
    metrics = {
        "frames_processed": frame_idx,
        "video_fps": round(fps, 3),
        "resolution": f"{width}x{height}",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "processing_fps": round(frame_idx / elapsed_seconds, 3) if elapsed_seconds else 0,
        "model_inference_seconds": round(inference_seconds, 3),
        "mean_inference_ms_per_frame": round(1000 * inference_seconds / frame_idx, 2) if frame_idx else 0,
        "detection_rows": len(rows),
        "mean_detections_per_frame": round(len(rows) / frame_idx, 3) if frame_idx else 0,
        "unique_track_ids": len(tracked_ids),
        "mean_confidence": round(sum(row["confidence"] for row in rows) / len(rows), 4) if rows else None,
        "clahe_enabled": clahe_enabled,
        "clahe_clip_limit": clahe_clip_limit if clahe_enabled else None,
    }
    return rows, fields, metrics


def classify_tracks(video_bytes: bytes, rows: list[dict], output_path: Path, batch_size: int = 8):
    """Classify one best-quality detected crop per track with Fishial's bundled TorchScript model."""
    import torch

    classifier_path = APP_DIR / "classifier_model" / "model.pt"
    if not classifier_path.exists():
        raise FileNotFoundError("Fishial species classifier weights are missing from classifier_model/model.pt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = torch.jit.load(str(classifier_path), map_location=device).eval()
    mapping_bytes = bytes(classifier.class_mapping_json_bytes.cpu().tolist())
    class_mapping = json.loads(mapping_bytes.decode("utf-8").rstrip("\x00"))

    rows_by_frame: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_frame.setdefault(int(row["frame"]), []).append(row)

    best_crops: dict[int, tuple[int, object]] = {}
    with tempfile.TemporaryDirectory(prefix="fishtrack-species-") as temp_dir:
        input_path = Path(temp_dir) / "input_video"
        input_path.write_bytes(video_bytes)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("Could not reopen the uploaded clip for species classification.")
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                for row in rows_by_frame.get(frame_idx, []):
                    track_id = row.get("track_id")
                    if track_id is None:
                        continue
                    x1, y1, x2, y2 = (int(round(row[key])) for key in ("x1", "y1", "x2", "y2"))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] * crop.shape[1] < 1200:
                        continue
                    area = crop.shape[0] * crop.shape[1]
                    if area > best_crops.get(int(track_id), (-1, None))[0]:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        resized = cv2.resize(rgb, (434, 154), interpolation=cv2.INTER_LINEAR)
                        best_crops[int(track_id)] = (area, resized)
                frame_idx += 1
        finally:
            cap.release()

        means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        stds = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        track_predictions = {}
        track_ids = list(best_crops)
        progress = st.progress(0.0, text="Classifying representative crops for tracked fish…")
        try:
            with torch.inference_mode():
                for start in range(0, len(track_ids), batch_size):
                    ids = track_ids[start:start + batch_size]
                    images = []
                    for track_id in ids:
                        rgb = best_crops[track_id][1]
                        tensor = torch.from_numpy(rgb.copy()).to(device).float() / 255.0
                        tensor = tensor.permute(2, 0, 1)
                        images.append((tensor - means[0]) / stds[0])
                    batch = torch.stack(images)
                    logits = classifier(batch)[1]
                    scores, classes = torch.topk(torch.softmax(logits, dim=1), k=3, dim=1)
                    for i, track_id in enumerate(ids):
                        top3 = []
                        for score, class_id in zip(scores[i].cpu().tolist(), classes[i].cpu().tolist()):
                            entry = class_mapping.get(str(int(class_id)), {})
                            top3.append({
                                "species": entry.get("label", f"Class {class_id}"),
                                "score": round(float(score), 4),
                                "species_id": entry.get("species_id"),
                            })
                        track_predictions[track_id] = top3
                    progress.progress(min((start + len(ids)) / max(len(track_ids), 1), 1.0), text=f"Classified {min(start + len(ids), len(track_ids))} of {len(track_ids)} tracks…")
        finally:
            progress.empty()

        # Re-render the clip with stable per-track species labels.
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError("Could not create the species-annotated MP4.")
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                for row in rows_by_frame.get(frame_idx, []):
                    track_id = row.get("track_id")
                    x1, y1, x2, y2 = (int(round(row[key])) for key in ("x1", "y1", "x2", "y2"))
                    color = (0, 220, 180)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    prediction = track_predictions.get(int(track_id), []) if track_id is not None else []
                    if prediction:
                        species = prediction[0]["species"]
                        score = prediction[0]["score"]
                        label = f"ID {track_id} | {species} {score:.2f}"
                        row["species_label"] = species
                        row["species_score"] = score
                        row["species_top3"] = json.dumps(prediction, ensure_ascii=False)
                    else:
                        label = f"ID {track_id} | fish" if track_id is not None else "fish"
                        row["species_label"] = ""
                        row["species_score"] = ""
                        row["species_top3"] = "[]"
                    cv2.putText(frame, label[:70], (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                writer.write(frame)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

    fields = list(rows[0].keys()) if rows else []
    csv_buffer = io.StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    csv_writer.writeheader()
    csv_writer.writerows(rows)
    return output_path.read_bytes(), csv_buffer.getvalue().encode("utf-8"), track_predictions, device


if page == "Upload" and st.button("Run fish detection and tracking", type="primary", disabled=not (st.session_state.get("weights_bytes") and st.session_state.get("video_bytes"))):
    with tempfile.TemporaryDirectory(prefix="fishtrack-") as temp_dir:
        temp = Path(temp_dir)
        model_name = st.session_state["weights_name"]
        video_name = st.session_state["video_name"]
        model_bytes = st.session_state["weights_bytes"]
        video_bytes = st.session_state["video_bytes"]
        model_upload_path = temp / ("model.zip" if model_name.lower().endswith(".zip") else "model.pt")
        video_path = temp / f"input_video{Path(video_name).suffix.lower()}"
        output_path = temp / "fishtrack_annotated.mp4"
        model_upload_path.write_bytes(model_bytes)
        video_path.write_bytes(video_bytes)

        try:
            model_path = get_model_path(model_upload_path, temp)
            try:
                model = YOLO(str(model_path), task="detect")
            except Exception as exc:
                raise ValueError(
                    "Could not load this checkpoint with Ultralytics. Upload the Fishial detector export (.pt/.torchscript), "
                    "not the classification model or training checkpoint. See the model repository for supported inference instructions."
                ) from exc

            rows, fields, metrics = process_video(model, video_path, output_path, confidence, clahe_enabled, clahe_clip_limit)
            csv_path = temp / "fish_tracks.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                csv_writer = csv.DictWriter(file, fieldnames=fields)
                csv_writer.writeheader()
                csv_writer.writerows(rows)

            video_bytes = output_path.read_bytes()
            csv_bytes = csv_path.read_bytes()
            metrics["model_file"] = model_name
            metrics["model_size_mb"] = round(len(model_bytes) / (1024 * 1024), 2)
            metrics["confidence_threshold"] = confidence
            metrics["note"] = "Runtime/output benchmark only; no accuracy metric without ground truth for this video."
            import json
            benchmark_bytes = json.dumps(metrics, indent=2).encode("utf-8")
            st.session_state.setdefault("fishtrack_benchmark_history", []).append({
                key: metrics[key] for key in (
                    "model_file", "model_size_mb", "frames_processed", "resolution",
                    "elapsed_seconds", "processing_fps", "mean_inference_ms_per_frame",
                    "detection_rows", "mean_detections_per_frame", "unique_track_ids",
                    "mean_confidence", "confidence_threshold",
                )
            })
            st.session_state["fishtrack_result"] = {
                "video": video_bytes,
                "csv": csv_bytes,
                "benchmark": benchmark_bytes,
                "metrics": metrics,
                "rows": rows,
            }
            st.success("Analysis finished. Open Results in the menu to review the annotated video and metrics.")
        except Exception as exc:
            st.error(str(exc))


result = st.session_state.get("fishtrack_result")
if page == "Results":
    st.markdown('<div class="section-title">Your analysis</div><div class="section-copy">Annotated video, runtime summary, and image-pixel trajectories.</div>', unsafe_allow_html=True)
    if not result:
        st.info("Run an analysis from Upload to see its output here.")
if page == "Results" and result:
    # Keep a normal local copy as a fallback when the browser blocks Streamlit downloads.
    export_dir = Path(__file__).parent / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    video_export_path = export_dir / "fishtrack_annotated.mp4"
    video_export_path.write_bytes(result["video"])
    st.success(f"Processed {result['metrics']['frames_processed']} frames · {result['metrics']['detection_rows']} detections · {result['metrics']['unique_track_ids']} unique track IDs")
    st.info("This is a speed/output benchmark, not an accuracy score. Track IDs can switch during occlusion. Coordinates and paths are image-pixel estimates, not calibrated distances.")
    metrics = result["metrics"]
    cols = st.columns(5)
    cols[0].metric("Processing speed", f"{metrics['processing_fps']} FPS")
    cols[1].metric("Model inference", f"{metrics['mean_inference_ms_per_frame']} ms/frame")
    cols[2].metric("Total runtime", f"{metrics['elapsed_seconds']} s")
    cols[3].metric("Mean detections/frame", metrics["mean_detections_per_frame"])
    cols[4].metric("Mean confidence", f"{metrics['mean_confidence']:.3f}" if metrics["mean_confidence"] is not None else "—")
    st.caption(f"Model: {metrics['model_file']} ({metrics['model_size_mb']} MB) · Video: {metrics['resolution']} at {metrics['video_fps']} FPS · Confidence threshold: {metrics['confidence_threshold']}")
    if st.button("Add Fishial species labels to this run", type="secondary", disabled=not st.session_state.get("video_bytes")):
        try:
            with tempfile.TemporaryDirectory(prefix="fishtrack-species-out-") as species_dir:
                species_video_path = Path(species_dir) / "fishtrack_species_annotated.mp4"
                species_video, species_csv, predictions, species_device = classify_tracks(
                    st.session_state["video_bytes"], result["rows"], species_video_path
                )
                result["video"] = species_video
                result["csv"] = species_csv
                result["species_count"] = len({p[0]["species"] for p in predictions.values() if p})
                result["species_track_count"] = len(predictions)
                result["species_device"] = species_device
                result["species_predictions"] = predictions
                st.session_state["fishtrack_result"] = result
                st.success(f"Added ranked species labels for {len(predictions)} tracks across {result['species_count']} predicted species.")
        except Exception as exc:
            st.error(f"Species classification failed: {exc}")
    if result.get("species_predictions"):
        st.caption(f"Fishial DinoV2 + ViT classifier · 866 supported classes · {result['species_device'].upper()} · stable per track from its largest detected crop. Labels are model candidates, not verified species IDs.")
        species_rows = [
            {"Track ID": track_id, "Candidate 1": pred[0]["species"], "Score": pred[0]["score"], "Candidate 2": pred[1]["species"], "Candidate 3": pred[2]["species"]}
            for track_id, pred in result["species_predictions"].items() if pred
        ]
        if species_rows:
            st.dataframe(pd.DataFrame(species_rows), use_container_width=True, hide_index=True)
    st.subheader("Annotated output")
    left, right = st.columns([2, 1])
    with left:
        st.video(result["video"])
    with right:
        st.download_button("Download annotated video", result["video"], "fishtrack_annotated.mp4", "video/mp4")
        st.caption(f"Local copy: `{video_export_path}`")
        st.download_button("Download track CSV", result["csv"], "fish_tracks.csv", "text/csv")
        st.download_button("Download benchmark JSON", result["benchmark"], "fishtrack_benchmark.json", "application/json")

    tracked_rows = [row for row in result["rows"] if row["track_id"] is not None]
    if tracked_rows:
        chart_data = pd.DataFrame(tracked_rows)
        trajectory = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X("center_x:Q", title="Image x (pixels)"),
            y=alt.Y("center_y:Q", title="Image y (pixels)", scale=alt.Scale(reverse=True)),
            color=alt.Color("track_id:N", title="Track ID"),
            detail="track_id:N",
            order="frame:Q",
            tooltip=["track_id:N", "frame:Q", "time_seconds:Q", "center_x:Q", "center_y:Q"],
        ).properties(height=360, title="Fish trajectories in image coordinates")
        st.altair_chart(trajectory, use_container_width=True)
    else:
        st.caption("No persistent track IDs were produced, so the trajectory plot is unavailable for this clip.")

history = st.session_state.get("fishtrack_benchmark_history", [])
if page == "Results" and history:
    st.subheader("Model comparison · runs in this session")
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

if page == "About":
    st.markdown('<div class="section-title">A clearer view beneath the surface</div><div class="section-copy">Seal is a local prototype for exploring underwater object detection and short-term tracking.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="info-card"><h3>How it works</h3><p>Load an Ultralytics YOLO fish detector, process an uploaded clip frame by frame, and associate detections with ByteTrack. Results include an annotated video, a CSV, benchmark timing, and image-coordinate trajectories.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-card"><h3>Read results carefully</h3><p>Track IDs are temporary and may switch when fish overlap or disappear. Coordinates and paths are pixel estimates, not real-world distances. Runtime and detection counts are not accuracy scores; evaluation needs labelled footage.</p></div>', unsafe_allow_html=True)
