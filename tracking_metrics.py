"""Tracking metrics for one MOTChallenge-format video sequence.

Scores are only meaningful when predictions and labelled tracks cover the same
frames, coordinate system, and ignore-region policy.
"""
from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict

import numpy as np


def _linear_sum_assignment(cost):
    """Rectangular Hungarian assignment, returning row/column NumPy arrays."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not cost.size:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    transposed = cost.shape[0] > cost.shape[1]
    matrix = cost.T if transposed else cost
    n, m = matrix.shape  # n <= m
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j] - 1), j - 1) for j in range(1, m + 1) if p[j] != 0]
    rows = np.asarray([pair[0] for pair in pairs], dtype=int)
    cols = np.asarray([pair[1] for pair in pairs], dtype=int)
    return (cols, rows) if transposed else (rows, cols)


def _parse_gt(text: str):
    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            raise ValueError(f"Ground-truth line {line_no} needs at least 6 comma-separated MOT fields.")
        try:
            frame, track = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 and parts[6] else 1.0
        except ValueError as exc:
            raise ValueError(f"Ground-truth line {line_no} contains a non-numeric MOT value.") from exc
        if frame < 1 or w <= 0 or h <= 0:
            raise ValueError(f"Ground-truth line {line_no} has an invalid frame or box size.")
        # MOTChallenge GT uses mark=0 for ignored / distractor annotations.
        if len(parts) > 6 and conf == 0:
            continue
        records.append((frame, track, (x, y, x + w, y + h)))
    if not records:
        raise ValueError("No labelled fish tracks were found in the ground-truth file.")
    return records


def _parse_predictions(text: str):
    records = []
    reader = csv.DictReader(io.StringIO(text))
    required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    if not reader.fieldnames or not required.issubset({name.strip() for name in reader.fieldnames}):
        raise ValueError("Prediction CSV must contain frame, track_id, x1, y1, x2, and y2 columns.")
    for line_no, row in enumerate(reader, 2):
        try:
            if row.get("track_id", "").strip().lower() in {"", "none", "null", "-1"}:
                continue
            # Seal's exported CSV is 0-based; MOT ground truth is 1-based.
            frame, track = int(float(row["frame"])) + 1, int(float(row["track_id"]))
            box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Prediction CSV line {line_no} has invalid frame, ID, or box coordinates.") from exc
        if frame < 1 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Prediction CSV line {line_no} has an invalid frame or box size.")
        records.append((frame, track, box))
    if not records:
        raise ValueError("No tracked predictions with valid IDs were found in this run.")
    return records


def _iou_matrix(gt_boxes, pred_boxes):
    if not gt_boxes or not pred_boxes:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
    g = np.asarray(gt_boxes, dtype=float)
    p = np.asarray(pred_boxes, dtype=float)
    left = np.maximum(g[:, None, 0], p[None, :, 0])
    top = np.maximum(g[:, None, 1], p[None, :, 1])
    right = np.minimum(g[:, None, 2], p[None, :, 2])
    bottom = np.minimum(g[:, None, 3], p[None, :, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    return intersection / np.maximum(area_g[:, None] + area_p[None, :] - intersection, 1e-12)


def evaluate_tracking(gt_text: str, prediction_csv: str):
    """Compute HOTA (0.05:0.95), IDF1, MOTA, and counts for one sequence."""
    gt = _parse_gt(gt_text)
    pred = _parse_predictions(prediction_csv)
    gt_frames, pred_frames = defaultdict(list), defaultdict(list)
    gt_track_totals, pred_track_totals = Counter(), Counter()
    for frame, identity, box in gt:
        gt_frames[frame].append((identity, box)); gt_track_totals[identity] += 1
    for frame, identity, box in pred:
        pred_frames[frame].append((identity, box)); pred_track_totals[identity] += 1
    frames = sorted(set(gt_frames) | set(pred_frames))

    # IDF1 uses one global identity assignment at the conventional IoU 0.5 threshold.
    idtp_pairs = Counter()
    mota_tp = mota_fp = mota_fn = id_switches = 0
    previous_match = {}
    frame_matches = {}
    for frame in frames:
        gs, ps = gt_frames[frame], pred_frames[frame]
        ious = _iou_matrix([g[1] for g in gs], [p[1] for p in ps])
        if ious.size:
            cost = np.where(ious >= 0.5, 1.0 - ious, 1e6)
            gi, pi = _linear_sum_assignment(cost)
            pairs = [(int(i), int(j)) for i, j in zip(gi, pi) if ious[i, j] >= 0.5]
        else:
            pairs = []
        frame_matches[frame] = (gs, ps, ious, pairs)
        mota_tp += len(pairs)
        mota_fn += len(gs) - len(pairs)
        mota_fp += len(ps) - len(pairs)
        for gi, pi in pairs:
            gid, pid = gs[gi][0], ps[pi][0]
            idtp_pairs[(gid, pid)] += 1
            if gid in previous_match and previous_match[gid] != pid:
                id_switches += 1
            previous_match[gid] = pid
    gt_ids, pred_ids = list(gt_track_totals), list(pred_track_totals)
    id_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    gt_index, pred_index = {x: i for i, x in enumerate(gt_ids)}, {x: i for i, x in enumerate(pred_ids)}
    for (gid, pid), count in idtp_pairs.items():
        id_matrix[gt_index[gid], pred_index[pid]] = count
    if id_matrix.size:
        ri, ci = _linear_sum_assignment(-id_matrix)
        idtp = float(id_matrix[ri, ci].sum())
    else:
        idtp = 0.0
    idfn = sum(gt_track_totals.values()) - idtp
    idfp = sum(pred_track_totals.values()) - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfn + idfp, 1e-12)
    mota = 1 - (mota_fn + mota_fp + id_switches) / max(len(gt), 1)

    hota_values = []
    for alpha in np.arange(0.05, 0.951, 0.05):
        tp = fp = fn = 0
        pair_tp = Counter()
        # Per-frame one-to-one IoU matching at alpha, as used by HOTA.
        for frame in frames:
            gs, ps, ious, _ = frame_matches[frame]
            if ious.size:
                cost = np.where(ious >= alpha, 1.0 - ious, 1e6)
                gi, pi = _linear_sum_assignment(cost)
                pairs = [(int(i), int(j)) for i, j in zip(gi, pi) if ious[i, j] >= alpha]
            else:
                pairs = []
            tp += len(pairs); fn += len(gs) - len(pairs); fp += len(ps) - len(pairs)
            for gi, pi in pairs:
                pair_tp[(gs[gi][0], ps[pi][0])] += 1
        deta = tp / max(tp + fp + fn, 1e-12)
        assa = 0.0
        if tp:
            assa = sum(count * count / max(gt_track_totals[gid] + pred_track_totals[pid] - count, 1)
                       for (gid, pid), count in pair_tp.items()) / tp
        hota_values.append((deta * assa) ** 0.5)
    return {
        "status": "scored",
        "hota": round(float(np.mean(hota_values)), 6),
        "idf1": round(float(idf1), 6),
        "mota": round(float(mota), 6),
        "hota_percent": round(float(np.mean(hota_values)) * 100, 2),
        "idf1_percent": round(float(idf1) * 100, 2),
        "mota_percent": round(float(mota) * 100, 2),
        "iou_threshold_idf1_mota": 0.5,
        "hota_iou_thresholds": [round(float(x), 2) for x in np.arange(0.05, 0.951, 0.05)],
        "ground_truth_detections": len(gt), "prediction_detections": len(pred),
        "true_positives_iou_0_5": int(mota_tp), "false_positives_iou_0_5": int(mota_fp),
        "false_negatives_iou_0_5": int(mota_fn), "identity_switches": int(id_switches),
        "frames_compared": len(frames),
        "note": "Computed against the uploaded MOTChallenge ground-truth file. HOTA is averaged over IoU thresholds 0.05–0.95; IDF1 and MOTA use IoU 0.50.",
    }
