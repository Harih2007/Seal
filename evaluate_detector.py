"""Compute validation-set detection metrics from a labeled Ultralytics dataset.

Example:
  python evaluate_detector.py --model weights/fish_detector.pt --data validation.yaml
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ultralytics detection checkpoint (.pt)")
    parser.add_argument("--data", required=True, help="Dataset YAML with a labeled val split")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001, help="Validation confidence cutoff (Ultralytics default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (Ultralytics default: 0.7)")
    parser.add_argument("--device", default=None, help="For example 0 for first CUDA GPU, or cpu")
    parser.add_argument("--output", default="validation_metrics.json")
    args = parser.parse_args()

    model_path, data_path = Path(args.model), Path(args.data)
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")
    if not data_path.is_file():
        parser.error(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path), task="detect")
    result = model.val(data=str(data_path), split="val", imgsz=args.imgsz, batch=args.batch, conf=args.conf, iou=args.iou,
                       device=args.device, plots=True, verbose=True)
    box = result.box
    report = {
        "model": str(model_path.resolve()), "dataset_yaml": str(data_path.resolve()),
        "split": "val", "imgsz": args.imgsz, "batch": args.batch, "confidence_cutoff": args.conf, "nms_iou": args.iou,
        "precision": float(box.mp), "recall": float(box.mr),
        "mAP50": float(box.map50), "mAP50_95": float(box.map),
        "note": "Ultralytics validation metrics on the dataset YAML val split. Confirm this split contains representative underwater challenge footage.",
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved metrics to {output_path.resolve()}")


if __name__ == "__main__":
    main()
