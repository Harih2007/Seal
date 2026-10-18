"""Fishial v10.2 species classification shared with the Streamlit prototype."""
import json
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from inference_engine import draw_track_trails, track_color
from video_encoding import create_video_writer


_CLASSIFIER_CACHE = {}
_CLASSIFIER_LOCK = __import__("threading").Lock()


def classify_tracks(video_path: Path, rows: list[dict], output_path: Path, progress_callback=None,
                    crop_candidates=None):
    """Classify several strong crops per track, then assign one stable track label."""
    import torch

    app_dir = Path(__file__).resolve().parent
    classifier_path = Path(os.environ.get("SEAL_SPECIES_MODEL", app_dir / "classifier_model" / "model.pt"))
    if not classifier_path.is_file():
        raise FileNotFoundError(
            "Fishial species classifier weights are missing from classifier_model/model.pt."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (str(classifier_path.resolve()), device)
    with _CLASSIFIER_LOCK:
        cached = _CLASSIFIER_CACHE.get(cache_key)
        if cached is None:
            try:
                classifier = torch.jit.load(str(classifier_path), map_location=device).eval()
                mapping_bytes = bytes(classifier.class_mapping_json_bytes.cpu().tolist())
                cached = (classifier, json.loads(mapping_bytes.decode("utf-8").rstrip("\x00")))
                _CLASSIFIER_CACHE[cache_key] = cached
            except Exception as exc:
                raise ValueError("Could not load the Streamlit Fishial species classifier bundle.") from exc
    classifier, class_mapping = cached

    rows_by_frame: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_frame.setdefault(int(row["frame"]), []).append(row)

    if crop_candidates is None:
        crop_candidates = defaultdict(list)
        cap = cv2.VideoCapture(str(video_path))
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
                    x1, y1, x2, y2 = (int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2"))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] * crop.shape[1] < 1200:
                        continue
                    area = crop.shape[0] * crop.shape[1]
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_candidates[int(track_id)].append((area, frame_idx,
                        cv2.resize(rgb, (434, 154), interpolation=cv2.INTER_LINEAR)))
                frame_idx += 1
        finally:
            cap.release()

    means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    # Keep up to five large crops spaced through the track. Averaging them avoids
    # a single blurred or occluded frame changing the species label.
    selected_crops: dict[int, list[np.ndarray]] = {}
    for track_id, candidates in crop_candidates.items():
        chosen, used_frames = [], []
        for area, candidate_frame, image in sorted(candidates, key=lambda item: item[0], reverse=True):
            if all(abs(candidate_frame - frame) >= 12 for frame in used_frames):
                chosen.append(image)
                used_frames.append(candidate_frame)
            if len(chosen) == 5:
                break
        if chosen:
            selected_crops[track_id] = chosen

    crop_predictions: dict[int, list[np.ndarray]] = defaultdict(list)
    flat_crops = [(track_id, image) for track_id, images in selected_crops.items() for image in images]
    classification_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(flat_crops), 16):
            batch = flat_crops[start:start + 16]
            # Convert and normalize a whole crop batch at once, avoiding one tensor
            # allocation and device transfer per crop while preserving the same inputs.
            image_array = np.stack([rgb for _track_id, rgb in batch]).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(0, 3, 1, 2).to(device)
            image_tensor = (image_tensor - means) / stds
            probabilities = torch.softmax(classifier(image_tensor)[1], dim=1).cpu().numpy()
            for i, (track_id, _image) in enumerate(batch):
                crop_predictions[track_id].append(probabilities[i])
            if progress_callback:
                progress_callback(min(start + len(batch), len(flat_crops)), len(flat_crops))
    classification_seconds = time.perf_counter() - classification_started

    track_predictions: dict[int, list[dict]] = {}
    track_decisions: dict[int, dict] = {}
    for track_id, probabilities in crop_predictions.items():
        averaged = np.mean(np.asarray(probabilities), axis=0)
        top_indices = np.argsort(averaged)[-3:][::-1]
        top3 = []
        for class_id in top_indices:
            entry = class_mapping.get(str(int(class_id)), {})
            top3.append({
                "species": entry.get("label", f"Class {class_id}"),
                "score": round(float(averaged[class_id]), 4),
                "species_id": entry.get("species_id"),
            })
        winner = int(top_indices[0])
        agreement = sum(int(np.argmax(vector)) == winner for vector in probabilities) / len(probabilities)
        confidence = float(averaged[winner])
        # A closed-set classifier cannot name an unseen species. Abstain when its
        # label changes across crops or its averaged confidence is weak.
        unlabelled = confidence < 0.40 or agreement < 0.60
        track_predictions[track_id] = top3
        track_decisions[track_id] = {
            "label": "Unlabelled fish" if unlabelled else top3[0]["species"],
            "score": round(confidence, 4),
            "unlabelled": unlabelled,
            "agreement": round(agreement, 3),
            "crops_used": len(probabilities),
            "candidate": top3[0]["species"],
        }

    # Store the same top-3 outputs in the track CSV as the Streamlit app.
    for row in rows:
        track_id = row.get("track_id")
        prediction = track_predictions.get(int(track_id), []) if track_id is not None else []
        decision = track_decisions.get(int(track_id), {}) if track_id is not None else {}
        row["species_label"] = decision.get("label", "")
        row["species_score"] = decision.get("score", "")
        row["species_unlabelled"] = decision.get("unlabelled", "")
        row["species_agreement"] = decision.get("agreement", "")
        row["species_top3"] = json.dumps(prediction, ensure_ascii=False)

    render_started = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not reopen the uploaded clip to render species labels.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = create_video_writer(output_path, fps, width, height)
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the species-annotated video.")
    frame_idx = 0
    trails = defaultdict(lambda: deque(maxlen=60))
    trail_last_seen = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rows = rows_by_frame.get(frame_idx, [])
            for row in frame_rows:
                track_id = row.get("track_id")
                if track_id is not None:
                    center = (int(round(float(row["center_x"]))), int(round(float(row["center_y"]))))
                    track_id = int(track_id)
                    trails[track_id].append(center)
                    trail_last_seen[track_id] = frame_idx
            frame = draw_track_trails(frame, trails, frame_idx, trail_last_seen)
            for row in frame_rows:
                track_id = row.get("track_id")
                x1, y1, x2, y2 = (int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2"))
                color = track_color(track_id) if track_id is not None else (0, 220, 180)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                decision = track_decisions.get(int(track_id), {}) if track_id is not None else {}
                if decision:
                    label = f"ID {track_id} | {decision['label']} {decision['score']:.2f}"
                else:
                    label = f"ID {track_id} | fish" if track_id is not None else "fish"
                cv2.putText(frame, label[:70], (max(0, x1), max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    annotation_render_seconds = time.perf_counter() - render_started

    counts = Counter(decision["label"] for decision in track_decisions.values())
    species_counts = [{"species": name, "track_count": count} for name, count in counts.most_common()]
    return {
        "track_predictions": track_predictions,
        "track_decisions": track_decisions,
        "species_counts": species_counts,
        "classified_tracks": len(track_predictions),
        "species_count": len(counts),
        "classifier": "Fishial v10.2 DinoV2 + ViT Pooling + ArcFace + KNN · 866 classes",
        "classifier_device": device,
        "video_encoder": writer.encoder,
        "crop_model_inference_seconds": round(classification_seconds, 3),
        "annotation_render_seconds": round(annotation_render_seconds, 3),
        "note": "One consensus label per track. Tracks with weak or inconsistent crop predictions are flagged as Unlabelled fish; that flags an out-of-dataset possibility, but does not name an unseen species.",
    }
