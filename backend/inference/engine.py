import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort

from backend.config import (
    MODEL_PATH,
    MODEL_CLASSES,
    MODEL_INPUT_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)
from backend.inference.preprocessing import letterbox, preprocess_tensor, generate_tiles
from backend.inference.postprocessing import decode_detections, apply_nms, unletterbox_boxes
from backend.inference.acoustic_physics import (
    classify_material_density,
    calculate_shadow_mensuration,
    calculate_threat_score,
    soft_nms
)

def compute_sha256(filepath: str) -> str:
    """Compute sha256 hash of file in read-only binary mode."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

class YOLOESIInferenceEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(YOLOESIInferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if self._initialized:
            return
        
        self.model_path = model_path or MODEL_PATH
        if not os.path.exists(self.model_path):
            print(f"[YOLO-ESI Engine] NOTICE: ONNX model not found at '{self.model_path}'.")
            print("[YOLO-ESI Engine] Initializing in Demonstration Mode. UI and API remain fully operational.")
            print("[YOLO-ESI Engine] To activate production ONNX inference, copy yolo_esi_fp16.onnx into models/")
            self.is_demo = True
            self.sha256_hash = "demo-simulation-mode"
            self.active_provider = "SimulationProvider"
            self.input_name = "images"
            self.input_shape = [1, 3, 256, 256]
            self.output_name = "output0"
            self.output_shape = [1, 8, 1344]
            self.session = None
        else:
            self.is_demo = False
            # Compute SHA256 checksum (Model is IMMUTABLE)
            self.sha256_hash = compute_sha256(self.model_path)
            
            # Select Execution Providers (GPU if available, CPU fallback)
            available_providers = ort.get_available_providers()
            selected_providers = []
            if "CUDAExecutionProvider" in available_providers:
                selected_providers.append("CUDAExecutionProvider")
            selected_providers.append("CPUExecutionProvider")

            # Create read-only session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=selected_providers
            )
            self.active_provider = self.session.get_providers()[0]

            # Inspect input & output tensors
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
            self.output_shape = self.session.get_outputs()[0].shape

        # Metadata
        self.model_name = "YOLOv8-ESI"
        self.classes = MODEL_CLASSES
        self.input_size = MODEL_INPUT_SIZE
        self.total_params = "3.3M parameters"
        self.architecture = "YOLOv8-Nano + Squeeze-and-Excitation (SE) Attention"
        
        self._initialized = True
        print(f"[YOLO-ESI Engine] Model initialized successfully from: {self.model_path}")
        print(f"[YOLO-ESI Engine] Provider: {self.active_provider} | SHA256: {self.sha256_hash[:16]}...")

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the active YOLO-ESI ONNX model."""
        return {
            "model_name": self.model_name,
            "format": "ONNX FP16" if "fp16" in self.model_path.lower() else "ONNX",
            "input_resolution": f"{self.input_size[0]}x{self.input_size[1]}",
            "execution_provider": self.active_provider,
            "model_path": self.model_path,
            "sha256_hash": self.sha256_hash,
            "classes": self.classes,
            "architecture": self.architecture,
            "attention_mechanism": "SE Channel Attention in C2f Feature Blocks"
        }

    def infer_single_image(
        self,
        img: np.ndarray,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        use_soft_nms: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs single-pass inference with letterboxing."""
        orig_shape = img.shape[:2]
        
        # Letterbox to model input size (256x256)
        letterboxed, ratio, pad = letterbox(img, new_shape=self.input_size, auto=False)
        tensor = preprocess_tensor(letterboxed)

        # Run ONNX inference or demo fallback
        if getattr(self, "is_demo", False):
            h, w = orig_shape
            cx, cy = w / 2.0, h / 2.0
            boxes = np.array([[cx - 35, cy - 25, cx + 35, cy + 25]], dtype=float)
            scores = np.array([0.885], dtype=float)
            class_ids = np.array([0], dtype=int)
            self._last_raw_output = np.zeros((1, 8, 1344), dtype=np.float32)
            return boxes, scores, class_ids

        raw_output = self.session.run([self.output_name], {self.input_name: tensor})[0]

        # Store raw output for heatmap generation
        self._last_raw_output = raw_output

        # Decode detections
        boxes_xyxy, scores, class_ids = decode_detections(raw_output, conf_threshold, self.classes)

        if len(boxes_xyxy) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        # Non-Maximum Suppression (Gaussian Soft-NMS or standard OpenCV NMS)
        if use_soft_nms:
            keep_indices = soft_nms(boxes_xyxy, scores, iou_threshold=iou_threshold)
        else:
            keep_indices = apply_nms(boxes_xyxy, scores, iou_threshold)
        if len(keep_indices) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        boxes_nms = boxes_xyxy[keep_indices]
        scores_nms = scores[keep_indices]
        class_ids_nms = class_ids[keep_indices]

        # Map back to original image space
        boxes_orig = unletterbox_boxes(boxes_nms, ratio, pad, orig_shape)

        return boxes_orig, scores_nms, class_ids_nms

    def infer_tiled(
        self,
        img: np.ndarray,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        tile_size: int = 256,
        overlap: int = 64,
        use_soft_nms: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs sliced / tiled inference for high resolution imagery and aggregates via global NMS."""
        tiles = generate_tiles(img, tile_size=tile_size, overlap=overlap)
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for tile in tiles:
            tile_img = tile["image"]
            x_off = tile["x_offset"]
            y_off = tile["y_offset"]

            boxes, scores, class_ids = self.infer_single_image(
                tile_img,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                use_soft_nms=use_soft_nms
            )

            if len(boxes) > 0:
                # Offset boxes to full image coordinates
                boxes[:, [0, 2]] += x_off
                boxes[:, [1, 3]] += y_off
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_class_ids.append(class_ids)

        # For small/medium imagery (<= 1024px), also run single-pass global resized inference to catch broader features.
        # For large imagery (> 1024px), bypass global pass to eliminate aspect-ratio quantization error
        # which can displace acoustic highlight crops onto bare seabed.
        h_full, w_full = img.shape[:2]
        if max(h_full, w_full) <= 1024:
            global_boxes, global_scores, global_classes = self.infer_single_image(
                img,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                use_soft_nms=use_soft_nms
            )
            if len(global_boxes) > 0:
                all_boxes.append(global_boxes)
                all_scores.append(global_scores)
                all_class_ids.append(global_classes)

        if len(all_boxes) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        merged_boxes = np.vstack(all_boxes)
        merged_scores = np.concatenate(all_scores)
        merged_class_ids = np.concatenate(all_class_ids)

        # Global NMS across all tiles (Gaussian Soft-NMS or standard OpenCV NMS)
        if use_soft_nms:
            keep_indices = soft_nms(merged_boxes, merged_scores, iou_threshold=iou_threshold)
        else:
            keep_indices = apply_nms(merged_boxes, merged_scores, iou_threshold)
        if len(keep_indices) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        return merged_boxes[keep_indices], merged_scores[keep_indices], merged_class_ids[keep_indices]

    def predict(
        self,
        img: np.ndarray,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        use_tiling: Optional[bool] = None,
        use_soft_nms: bool = True,
        raw_gray: Optional[np.ndarray] = None,
        towfish_altitude_m: Optional[float] = None,
        pixel_resolution: Optional[Tuple[float, float]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Executes YOLO-ESI inference and returns structured detection objects and timing metrics.
        Computes acoustic backscatter material density (P95) and shadow mensuration.
        """
        t0 = time.perf_counter()
        h, w = img.shape[:2]

        # Prepare un-normalized raw grayscale array for acoustic physics analysis
        if raw_gray is None:
            if len(img.shape) == 2:
                raw_gray = img
            elif img.shape[2] == 4:
                raw_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                raw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Auto-detect if tiling is beneficial (image larger than 512x512)
        if use_tiling is None:
            use_tiling = (max(h, w) > 512)

        t_inf_start = time.perf_counter()
        if use_tiling:
            boxes, scores, class_ids = self.infer_tiled(
                img,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                use_soft_nms=use_soft_nms
            )
        else:
            boxes, scores, class_ids = self.infer_single_image(
                img,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                use_soft_nms=use_soft_nms
            )
        t_inf_end = time.perf_counter()

        inference_time_ms = round((t_inf_end - t_inf_start) * 1000.0, 2)
        
        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            score = float(scores[i])
            cid = int(class_ids[i])
            cname = self.classes.get(cid, "marine_debris")

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            bbox_dict = {
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2)
            }

            # Defense-grade Acoustic Physics Analysis
            mat_info = classify_material_density(raw_gray, bbox_dict)
            mens_info = calculate_shadow_mensuration(
                raw_gray,
                bbox_dict,
                towfish_altitude_m=towfish_altitude_m,
                pixel_resolution_m=pixel_resolution
            )

            threat_score = calculate_threat_score(
                material_density=mat_info["material_density"],
                estimated_height_meters=mens_info["estimated_height_meters"],
                confidence=score,
                class_name=cname
            )

            detections.append({
                "id": i + 1,
                "class_id": cid,
                "class_name": cname,
                "confidence": round(score, 4),
                "bbox": bbox_dict,
                "center_pixel": {
                    "x": round(float(cx), 2),
                    "y": round(float(cy), 2)
                },
                "material_density": mat_info["material_density"],
                "peak_backscatter_p95": mat_info["peak_backscatter_p95"],
                "estimated_height_meters": mens_info["estimated_height_meters"],
                "target_length_meters": mens_info.get("target_length_meters"),
                "target_width_meters": mens_info.get("target_width_meters"),
                "shadow_length_meters": mens_info["shadow_length_meters"],
                "threat_score": threat_score,
                "acoustic_telemetry": {
                    "mean_backscatter": mat_info["mean_backscatter"],
                    "reflectivity_ratio": mat_info["reflectivity_ratio"],
                    "impedance_estimate_mrayl": mat_info["impedance_estimate_mrayl"],
                    "classification_confidence": mat_info["classification_confidence"],
                    "shadow_detected": mens_info["shadow_detected"],
                    "grazing_angle_deg": mens_info["grazing_angle_deg"],
                    "mensuration_method": mens_info["mensuration_method"],
                    "density_description": mat_info["description"]
                }
            })

        # Sort detections by threat score descending (with confidence tie-breaker)
        detections.sort(key=lambda d: (d.get("threat_score", 0), d["confidence"]), reverse=True)
        # Re-index IDs sequentially
        for idx, det in enumerate(detections):
            det["id"] = idx + 1

        total_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        timing = {
            "inference_time_ms": inference_time_ms,
            "total_time_ms": total_time_ms
        }

        return detections, timing
