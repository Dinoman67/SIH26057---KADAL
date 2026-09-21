"""
backend/inference/acoustic_physics.py

Acoustic Physics & Hydrographic Mensuration Engine for SonarVision.
Implements:
1. Target Material Density Classification via Peak Acoustic Backscatter (95th Percentile Intensity, P95)
   - Specific Acoustic Impedance Z = rho * c
   - Specular Reflectivity: High-Z Metals (Z ~ 46 MRayl) vs Low-Z Synthetics/Plastics (Z ~ 2.5 MRayl)
2. Defense-Grade Target Mensuration via Acoustic Shadow Trigonometry
   - Shadow-based relief calculation: H_t = (H_tow * L_s) / (R_s + L_s) or H_t = L_s * tan(theta_g)
3. Gaussian Soft-NMS for Dense Acoustic Debris Fields
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# Standard acoustic backscatter thresholds on 8-bit scale [0, 255]
# 95th percentile intensity filters out single-pixel Rayleigh speckle spikes
P95_METALLIC_THRESHOLD = 185.0
SEABED_AMBIENT_PERCENTILE = 35.0
SHADOW_INTENSITY_THRESHOLD = 30.0

# Material Labels
MATERIAL_HARD_METAL = "Hard (Metallic)"
MATERIAL_SOFT_SYNTHETIC = "Soft (Synthetic/Plastic)"
MATERIAL_INDETERMINATE = "Unclassified Benthic Target"


def classify_material_density(
    raw_gray_image: np.ndarray,
    bbox: Dict[str, float]
) -> Dict[str, Any]:
    """
    Classifies material density (Hard/Metallic vs. Soft/Synthetic Plastic) by measuring
    the Peak Acoustic Backscatter (95th percentile intensity, P95) within the detection bounding box.

    Acoustic Physics Rationale:
    - Acoustic Impedance Z = rho * c:
      * Seawater: Z ~ 1.54 MRayl
      * High-density Ferrous / Non-ferrous Metals (Steel, Iron, Brass, Aluminum):
        Z ~ 40 - 47 MRayl -> Reflection Coefficient R = ((Z2 - Z1)/(Z2 + Z1))^2 ~ 0.85 - 0.88.
        Produces saturated, specular acoustic highlights.
      * Synthetic Polymers, Plastics, Monofilament Nylon (Ghost Nets), Polyethylene:
        Z ~ 2.0 - 3.2 MRayl -> Reflection Coefficient R ~ 0.03 - 0.08.
        Produces muted, diffuse backscatter with low peak reflectivity.
    - P95 Metric:
      Side-scan sonar imagery suffers from multiplicative acoustic speckle noise (Rayleigh / K-distribution).
      Using the 95th percentile (P95) instead of P100 (maximum) filters single-pixel speckle spike
      outliers while reliably capturing the true specular reflection front.
    """
    h, w = raw_gray_image.shape[:2]
    x1 = max(0, int(round(bbox.get("x1", 0))))
    y1 = max(0, int(round(bbox.get("y1", 0))))
    x2 = min(w, int(round(bbox.get("x2", w))))
    y2 = min(h, int(round(bbox.get("y2", h))))

    if x2 <= x1 or y2 <= y1:
        return {
            "material_density": MATERIAL_INDETERMINATE,
            "peak_backscatter_p95": 0.0,
            "mean_backscatter": 0.0,
            "reflectivity_ratio": 0.0,
            "impedance_estimate_mrayl": 1.54,
            "classification_confidence": 0.5,
            "description": "Degenerate bounding box geometry"
        }

    # Extract crop patch
    patch = raw_gray_image[y1:y2, x1:x2]
    if len(patch.shape) == 3:
        # Convert BGR/RGB to grayscale if needed
        patch = np.dot(patch[..., :3], [0.114, 0.587, 0.299])

    # Convert to float array in 0-255 range
    if patch.dtype == np.uint8:
        patch_float = patch.astype(np.float32)
    elif patch.dtype in [np.uint16, np.int16]:
        patch_float = (patch.astype(np.float32) / 65535.0) * 255.0
    elif np.issubdtype(patch.dtype, np.floating):
        if patch.max() <= 1.0:
            patch_float = patch * 255.0
        else:
            patch_float = patch
    else:
        patch_float = patch.astype(np.float32)

    # Compute backscatter statistics using Adaptive Backscatter Window:
    # Isolate top 30% brightest pixels first to eliminate seabed dilution in loose boxes
    if patch_float.size > 0:
        threshold_70 = float(np.percentile(patch_float, 70))
        bright_cluster = patch_float[patch_float >= threshold_70]
        if bright_cluster.size > 0:
            p95 = float(np.percentile(bright_cluster, 95))
        else:
            p95 = float(np.percentile(patch_float, 95))
    else:
        p95 = 0.0

    mean_val = float(np.mean(patch_float)) if patch_float.size > 0 else 0.0
    median_val = float(np.median(patch_float)) if patch_float.size > 0 else 0.0
    p99 = float(np.percentile(patch_float, 99)) if patch_float.size > 0 else 0.0

    # Normalized reflectivity ratio [0.0 - 1.0]
    reflectivity_ratio = round(min(1.0, max(0.0, p95 / 255.0)), 4)

    # Classification logic based on acoustic impedance thresholds
    if p95 >= P95_METALLIC_THRESHOLD:
        material = MATERIAL_HARD_METAL
        # Estimate acoustic impedance Z ~ 35-46 MRayl
        impedance_est = round(35.0 + (p95 - P95_METALLIC_THRESHOLD) / (255.0 - P95_METALLIC_THRESHOLD) * 12.0, 1)
        conf = round(min(0.99, 0.70 + (p95 - P95_METALLIC_THRESHOLD) / (255.0 - P95_METALLIC_THRESHOLD) * 0.29), 3)
        desc = f"Specular highlight detected (P95={p95:.1f}/255). Acoustic impedance Z ~ {impedance_est} MRayl indicates dense metallic structure."
    else:
        material = MATERIAL_SOFT_SYNTHETIC
        # Estimate acoustic impedance Z ~ 2.0-8.0 MRayl
        impedance_est = round(2.0 + (p95 / P95_METALLIC_THRESHOLD) * 6.0, 1)
        conf = round(min(0.95, 0.65 + (P95_METALLIC_THRESHOLD - p95) / P95_METALLIC_THRESHOLD * 0.30), 3)
        desc = f"Diffuse acoustic return (P95={p95:.1f}/255). Acoustic impedance Z ~ {impedance_est} MRayl indicates low-density synthetic polymer or textile."

    return {
        "material_density": material,
        "peak_backscatter_p95": round(p95, 2),
        "mean_backscatter": round(mean_val, 2),
        "median_backscatter": round(median_val, 2),
        "p99_backscatter": round(p99, 2),
        "reflectivity_ratio": reflectivity_ratio,
        "impedance_estimate_mrayl": impedance_est,
        "classification_confidence": conf,
        "description": desc
    }


def calculate_shadow_mensuration(
    raw_gray_image: np.ndarray,
    bbox: Dict[str, float],
    towfish_altitude_m: Optional[float] = None,
    pixel_resolution_m: Optional[Tuple[float, float]] = None,
    slant_range_m: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes hydrographic target mensuration (estimated target relief / height above seafloor)
    using acoustic shadow geometry. Supports bimodal shadow detection (internal vs. external).

    Hydrographic Formula:
        H_target = (H_towfish * L_shadow) / (R_slant + L_shadow)
        or via nominal grazing model:
        H_target = L_shadow * tan(theta_nominal)
    """
    h, w = raw_gray_image.shape[:2]
    x1 = max(0, int(round(bbox.get("x1", 0))))
    y1 = max(0, int(round(bbox.get("y1", 0))))
    x2 = min(w, int(round(bbox.get("x2", w))))
    y2 = min(h, int(round(bbox.get("y2", h))))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Resolution default: 0.5 m/px if GeoTIFF NOAA standard, else 0.25 m/px nominal
    if pixel_resolution_m and len(pixel_resolution_m) >= 2:
        res_x = abs(float(pixel_resolution_m[0]))
        res_y = abs(float(pixel_resolution_m[1]))
    else:
        res_x = 0.5
        res_y = 0.5

    # Extract patch inside bounding box
    patch = raw_gray_image[y1:y2, x1:x2]
    if len(patch.shape) == 3:
        patch = np.dot(patch[..., :3], [0.114, 0.587, 0.299])

    shadow_px = 0
    shadow_detected = False

    # 1. Bimodal Check: Test if acoustic shadow is already INSIDE bounding box (Kaggle/large target style)
    if patch.size > 0:
        p10 = float(np.percentile(patch, 10))
        if p10 < SHADOW_INTENSITY_THRESHOLD:
            # Internal shadow detected: compute horizontal width of dark cluster (intensity < 30)
            dark_mask = (patch < SHADOW_INTENSITY_THRESHOLD)
            cols_with_shadow = int(np.sum(np.any(dark_mask, axis=0)))
            if cols_with_shadow >= 2:
                shadow_px = cols_with_shadow
                shadow_detected = True

    # Helper to measure shadow contiguous run
    def measure_strip_shadow(strip: np.ndarray, from_start: bool = True) -> int:
        if strip.size == 0:
            return 0
        if len(strip.shape) == 3:
            strip = np.dot(strip[..., :3], [0.114, 0.587, 0.299])
        profile = np.mean(strip, axis=0 if strip.shape[0] == (y2 - y1) else 1)
        if not from_start:
            profile = profile[::-1]
        
        ambient = np.percentile(raw_gray_image, SEABED_AMBIENT_PERCENTILE)
        cutoff = max(15.0, min(SHADOW_INTENSITY_THRESHOLD, ambient * 0.45))
        
        shadow_pixels = 0
        for val in profile:
            if val <= cutoff:
                shadow_pixels += 1
            elif shadow_pixels > 0:
                break
        return shadow_pixels

    # 2. If no internal shadow exists (NOAA-style tight highlight annotation), search horizontally outside x2
    if not shadow_detected:
        search_length_px = min(max(bw, bh) * 4, 150)
        right_x2 = min(w, x2 + search_length_px)
        right_strip = raw_gray_image[y1:y2, x2:right_x2]

        left_x1 = max(0, x1 - search_length_px)
        left_strip = raw_gray_image[y1:y2, left_x1:x1]

        shadow_right = measure_strip_shadow(right_strip, from_start=True)
        shadow_left = measure_strip_shadow(left_strip, from_start=False)

        if shadow_right >= 2 or shadow_left >= 2:
            shadow_px = max(shadow_right, shadow_left)
            shadow_detected = True
        else:
            # Minimum physical shadow based on target footprint aspect
            shadow_px = max(2, int(round(min(bw, bh) * 0.5)))
            shadow_detected = False

    shadow_m = round(shadow_px * res_x, 2)

    # Mensuration calculation:
    # If towfish altitude and slant range are available (e.g. from native XTF ping headers)
    if towfish_altitude_m and float(towfish_altitude_m) > 0.0:
        alt = float(towfish_altitude_m)
        slant = float(slant_range_m) if (slant_range_m and float(slant_range_m) > 0.0) else (alt * 3.5)
        # H_t = (H_tow * L_s) / (R_s + L_s)
        estimated_height = (alt * shadow_m) / (slant + shadow_m)
        grazing_deg = round(float(np.degrees(np.arctan2(alt, slant))), 1)
        method = "Towfish Altitude Trigonometry (Strict Hydrographic)"
    else:
        # Standard nominal grazing angle for SSS orthomosaics (15 deg nominal)
        nominal_grazing_rad = np.radians(15.0)
        estimated_height = shadow_m * np.tan(nominal_grazing_rad)
        grazing_deg = 15.0
        method = "Nominal Hydrographic Grazing Model (15°)"

    estimated_height = round(float(np.clip(estimated_height, 0.15, 35.0)), 2)

    # Calculate real-world physical target footprint dimensions (L x W)
    dim_x_m = round(bw * res_x, 2)
    dim_y_m = round(bh * res_y, 2)
    target_length_m = max(dim_x_m, dim_y_m)
    target_width_m = min(dim_x_m, dim_y_m)

    return {
        "estimated_height_meters": estimated_height,
        "target_length_meters": target_length_m,
        "target_width_meters": target_width_m,
        "shadow_length_meters": shadow_m,
        "shadow_length_pixels": shadow_px,
        "shadow_detected": shadow_detected,
        "grazing_angle_deg": grazing_deg,
        "mensuration_method": method
    }


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    sigma: float = 0.5,
    min_score: float = 0.05
) -> List[int]:
    """
    Gaussian Soft-NMS implementation for dense acoustic debris fields.
    Instead of completely eliminating overlapping candidate boxes, it decays
    their confidence scores using a Gaussian penalty:
        s_i = s_i * exp(- iou^2 / sigma)
    Preserves adjacent fragmented wreckage components that standard greedy NMS discards.
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.copy()
    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores.copy()

    areas = (x2 - x1) * (y2 - y1)
    n = len(boxes)
    indices = np.arange(n)
    keep = []

    for i in range(n):
        # Pick box with current maximum score
        max_idx = i + np.argmax(s[i:])
        # Swap
        s[i], s[max_idx] = s[max_idx], s[i]
        indices[i], indices[max_idx] = indices[max_idx], indices[i]
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        areas[[i, max_idx]] = areas[[max_idx, i]]

        if s[i] < min_score:
            break
        keep.append(int(indices[i]))

        # Calculate IoU with remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])

        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter
        iou = inter / (areas[i] + areas[i+1:] - inter + 1e-7)

        # Apply Gaussian decay for boxes exceeding threshold
        decay = np.exp(-(iou ** 2) / sigma)
        s[i+1:] = s[i+1:] * decay

    return keep


def calculate_threat_score(
    material_density: Optional[str] = None,
    estimated_height_meters: Optional[float] = None,
    confidence: float = 0.5,
    class_name: Optional[str] = None
) -> int:
    """
    Computes automated tactical threat score (0-100 integer) for naval C2 triage:
    - Material Multiplier: High Density/Metal = 3.0, Medium = 2.0, Low/Synthetic = 1.0
    - Height Multiplier: Min(estimated_height_meters, 5.0) or default to 1.0 if null
    - Formula: Normalize (Material Multiplier * Height Multiplier * Confidence * 33.3) to a max of 100
    """
    mat_str = (material_density or "").lower()
    if "hard" in mat_str or "metal" in mat_str:
        mat_mult = 3.0
    elif "soft" in mat_str or "synthetic" in mat_str or "plastic" in mat_str:
        mat_mult = 1.0
    else:
        mat_mult = 2.0

    if estimated_height_meters is not None and float(estimated_height_meters) > 0.0:
        height_mult = min(float(estimated_height_meters), 5.0)
    else:
        height_mult = 1.0

    conf = max(0.0, min(1.0, float(confidence)))
    raw_score = mat_mult * height_mult * conf * 33.3
    return int(round(min(100.0, max(0.0, raw_score))))

