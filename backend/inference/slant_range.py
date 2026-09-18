"""
backend/inference/slant_range.py

Hydrographic Slant Range Correction (SRC) Engine for Side-Scan Sonar Imagery.
Corrects geometric compression and removes water column nadir artifacts:
    R_ground = sqrt(R_slant^2 - H_towfish^2)
"""

import numpy as np
from typing import Tuple, Optional


def detect_first_bottom_return(
    ping: np.ndarray,
    min_search_ratio: float = 0.02,
    max_search_ratio: float = 0.40
) -> int:
    """
    Detects the first bottom return (nadir strike) index in a sonar ping scanline.
    Uses gradient edge detection and thresholding across the water column.
    """
    n = len(ping)
    if n == 0:
        return 0

    start_idx = max(1, int(n * min_search_ratio))
    end_idx = min(n - 1, int(n * max_search_ratio))

    if end_idx <= start_idx:
        return int(n * 0.1)

    window = ping[start_idx:end_idx].astype(np.float32)
    # Smooth with simple moving average
    kernel_size = max(3, int(n * 0.005))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(window, kernel, mode="same")

    # Compute first derivative
    diff = np.diff(smoothed)
    if len(diff) == 0:
        return start_idx

    # Pick index with maximum gradient above mean ambient water column level
    peak_diff_idx = int(np.argmax(diff))
    detected_idx = start_idx + peak_diff_idx
    return max(1, detected_idx)


def correct_slant_range_scanline(
    scanline: np.ndarray,
    altitude_px: int
) -> np.ndarray:
    """
    Applies slant-range to ground-range geometric mapping to a single side-scan half-swath scanline:
        R_ground = sqrt(R_slant^2 - H^2)
    """
    n_slant = len(scanline)
    if altitude_px <= 0 or altitude_px >= n_slant - 1:
        return scanline

    n_ground = int(np.sqrt(max(1, n_slant**2 - altitude_px**2)))
    ground_indices = np.arange(n_ground, dtype=np.float32)

    # R_slant = sqrt(R_ground^2 + H^2)
    slant_indices = np.sqrt(ground_indices**2 + float(altitude_px)**2)
    slant_indices = np.clip(slant_indices, 0, n_slant - 1)

    # Linear interpolation
    idx_floor = slant_indices.astype(np.int32)
    idx_ceil = np.clip(idx_floor + 1, 0, n_slant - 1)
    weight = slant_indices - idx_floor

    corrected = (1.0 - weight) * scanline[idx_floor] + weight * scanline[idx_ceil]
    return corrected.astype(scanline.dtype)


def correct_slant_range_waterfall(
    waterfall: np.ndarray,
    altitude_px: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Performs full swath Slant Range Correction on a 2D sonar waterfall mosaic.
    Assumes nadir is at the center column (port on left, starboard on right), or along axis 1.
    """
    if len(waterfall.shape) == 3:
        gray = np.dot(waterfall[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    else:
        gray = waterfall.copy()

    h, w = gray.shape[:2]
    half_w = w // 2

    # Port and Starboard channels
    port = gray[:, :half_w][:, ::-1]  # flip horizontally so nadir is at index 0
    starboard = gray[:, half_w:]       # nadir is at index 0

    # Auto-detect altitude if not provided
    if altitude_px is None or altitude_px <= 0:
        # Detect median altitude across pings
        sample_pings = starboard[::max(1, h // 20)]
        detected_alts = [detect_first_bottom_return(p) for p in sample_pings]
        altitude_px = int(np.median(detected_alts)) if detected_alts else max(5, int(half_w * 0.1))

    # Clamp altitude to avoid degenerate dimensions
    altitude_px = max(1, min(altitude_px, int(half_w * 0.75)))

    corrected_port_lines = []
    corrected_stbd_lines = []

    for row in range(h):
        c_port = correct_slant_range_scanline(port[row], altitude_px)
        c_stbd = correct_slant_range_scanline(starboard[row], altitude_px)
        corrected_port_lines.append(c_port)
        corrected_stbd_lines.append(c_stbd)

    corrected_port = np.array(corrected_port_lines)[:, ::-1]  # flip back
    corrected_stbd = np.array(corrected_stbd_lines)

    # Harmonize widths
    min_w = min(corrected_port.shape[1], corrected_stbd.shape[1])
    corrected_full = np.hstack([corrected_port[:, -min_w:], corrected_stbd[:, :min_w]])

    return corrected_full, altitude_px
