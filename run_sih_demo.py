#!/usr/bin/env python3
"""
SIH 2026 Demo Test Runner — KADAL YOLOv8-ESI
===================================================
Runs verified inference across all SIH_demo images, displays formatted benchmark
tables with exact detection confidences, and produces annotated validation crops.
"""

import os
import json
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_DIR = PROJECT_ROOT / "SIH_demo"
OUTPUT_DIR = DEMO_DIR / "annotated"
MANIFEST_PATH = DEMO_DIR / "manifest.json"

MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "yolo_esi_v6_fp16.onnx",
    PROJECT_ROOT / "models" / "yolo_esi_fp16.onnx",
    PROJECT_ROOT / "models" / "yolo_esi_v6_fp32.onnx",
]

CLASSES = {
    0: "unknown_debris",
    1: "airplane",
    2: "mine",
    3: "wreck"
}

CLASS_COLORS = {
    0: (0, 165, 255),    # Orange for marine debris
    1: (255, 191, 0),    # Cyan/Sky Blue for airplane
    2: (0, 0, 255),      # Red for naval mine
    3: (0, 255, 0)       # Green for shipwreck
}

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 256


def letterbox(img, target_size=256):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx, dy = (target_size - nw) // 2, (target_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas, scale, dx, dy


def run_demo():
    model_path = None
    for cand in MODEL_CANDIDATES:
        if cand.exists():
            model_path = str(cand)
            break

    if not model_path:
        raise FileNotFoundError("Could not find YOLOv8-ESI ONNX model in models/")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 95)
    print("🌊 KADAL: SIH 2026 JUDGES LIVE DEMO TEST SUITE")
    print(f"Model: {Path(model_path).name} (YOLOv8-ESI Edge Engine)")
    print(f"Confidence Threshold: {CONF_THRESHOLD} | NMS IoU: {IOU_THRESHOLD} | Resolution: 256x256")
    print("═" * 95 + "\n")

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    print(f"{'#':<3} {'Filename':<35} {'Category':<15} {'Detections':<12} {'Max Conf':<10} {'Latency':<9} {'Status'}")
    print("─" * 95)

    for item in manifest:
        idx = item["index"]
        fn = item["filename"]
        cat = item["category"]
        img_path = DEMO_DIR / fn

        if not img_path.exists():
            print(f"Warning: {fn} not found in {DEMO_DIR}")
            continue

        raw_img = cv2.imread(str(img_path))
        orig_h, orig_w = raw_img.shape[:2]

        t0 = time.perf_counter()
        canvas, scale, dx, dy = letterbox(raw_img, INPUT_SIZE)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

        outputs = session.run([output_name], {input_name: tensor})[0]
        t_ms = (time.perf_counter() - t0) * 1000.0

        predictions = np.squeeze(outputs).T

        boxes, scores, class_ids = [], [], []
        for row in predictions:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf > CONF_THRESHOLD:
                cx, cy, bw, bh = row[:4]
                x1 = max(0, int(((cx - bw / 2) - dx) / scale))
                y1 = max(0, int(((cy - bh / 2) - dy) / scale))
                x2 = min(orig_w, int(((cx + bw / 2) - dx) / scale))
                y2 = min(orig_h, int(((cy + bh / 2) - dy) / scale))
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(conf)
                class_ids.append(cls_id)

        nms_indices = []
        if len(boxes) > 0:
            nms_indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

        annotated = raw_img.copy()
        max_conf = 0.0

        if len(nms_indices) > 0:
            for nms_idx in nms_indices:
                i = nms_idx if isinstance(nms_idx, (int, np.integer)) else nms_idx[0]
                bx, by, bw, bh = boxes[i]
                cid = class_ids[i]
                sc = scores[i]
                cname = CLASSES.get(cid, str(cid))
                color = CLASS_COLORS.get(cid, (0, 255, 0))

                if sc > max_conf:
                    max_conf = sc

                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)
                label = f"{cname} {sc*100:.1f}%"
                cv2.putText(annotated, label, (bx, max(by - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        out_img_path = OUTPUT_DIR / f"annotated_{fn}"
        cv2.imwrite(str(out_img_path), annotated)

        n_dets = len(nms_indices) if len(nms_indices) > 0 else 0
        status = "✅ PASS"
        if cat == "clean" and n_dets == 0:
            status = "✅ 0 FALSE ALARMS"
        elif cat == "clean" and n_dets > 0:
            status = "❌ FALSE ALARM"
        elif n_dets == 0 and cat != "clean":
            status = "❌ MISS"

        conf_str = f"{max_conf*100:.1f}%" if n_dets > 0 else "N/A"
        print(f"{idx:<3} {fn:<35} {cat:<15} {n_dets:<12} {conf_str:<10} {t_ms:>5.1f} ms   {status}")

    print("─" * 95)
    print(f"✅ Annotated verification images successfully saved to: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    run_demo()
