"""
backend/inference/xtf_parser.py

Defense-Grade Native eXtended Triton Format (.XTF) Sonar Ingestion Engine.
Parses raw high-frequency / low-frequency side-scan sonar waterfall data,
transducer navigation telemetry, and towfish altitude via pyxtf.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    import pyxtf
    HAS_PYXTF = True
except ImportError:
    HAS_PYXTF = False

from backend.inference.slant_range import correct_slant_range_waterfall


def parse_xtf_file(
    filepath: str,
    apply_slant_correction: bool = True
) -> Dict[str, Any]:
    """
    Parses an eXtended Triton Format (.XTF) sonar file into calibrated waterfall arrays
    and hydrographic navigation telemetry.
    """
    if not HAS_PYXTF:
        raise RuntimeError("pyxtf library is required to ingest .XTF files. Run: pip install pyxtf")

    path_str = str(filepath)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"XTF file not found at: {path_str}")

    # Read XTF file header and packets
    file_header, packets = pyxtf.xtf_read(path_str)

    # Extract sonar ping headers
    sonar_packet_key = pyxtf.XTFHeaderType.sonar
    pings = packets.get(sonar_packet_key, [])

    if not pings:
        raise ValueError("XTF file contains no raw sidescan sonar ping packets (XTFHeaderType.sonar).")

    num_pings = len(pings)
    num_channels = file_header.NumberOfSonarChannels

    # Extract telemetry metrics across pings
    altitudes = []
    depths = []
    headings = []
    pitches = []
    rolls = []
    ship_x = []
    ship_y = []

    for p in pings:
        if hasattr(p, "SensorAltitude") and p.SensorAltitude > 0:
            altitudes.append(float(p.SensorAltitude))
        if hasattr(p, "SensorDepth") and p.SensorDepth > 0:
            depths.append(float(p.SensorDepth))
        if hasattr(p, "SensorHeading"):
            headings.append(float(p.SensorHeading))
        if hasattr(p, "SensorPitch"):
            pitches.append(float(p.SensorPitch))
        if hasattr(p, "SensorRoll"):
            rolls.append(float(p.SensorRoll))
        if hasattr(p, "ShipXcoordinate") and p.ShipXcoordinate != 0:
            ship_x.append(float(p.ShipXcoordinate))
        if hasattr(p, "ShipYcoordinate") and p.ShipYcoordinate != 0:
            ship_y.append(float(p.ShipYcoordinate))

    mean_altitude = float(np.median(altitudes)) if altitudes else None
    mean_depth = float(np.mean(depths)) if depths else None
    mean_heading = float(np.mean(headings)) if headings else None
    mean_pitch = float(np.mean(pitches)) if pitches else 0.0
    mean_roll = float(np.mean(rolls)) if rolls else 0.0

    # Sensor positions (WGS84 Lat/Lon or projected coordinates)
    lat_val = float(np.mean(ship_y)) if ship_y else None
    lon_val = float(np.mean(ship_x)) if ship_x else None

    # Concatenate sonar channels
    # Channel 0: Port, Channel 1: Starboard
    chan_arrays = []
    for ch in range(min(num_channels, 2)):
        try:
            arr = pyxtf.concatenate_channel(pings, file_header, channel=ch, weighted=False)
            chan_arrays.append(arr)
        except Exception as e:
            logger.warning("[XTF Parser] Failed to concatenate channel %s: %s", ch, e)

    if not chan_arrays:
        raise ValueError("Failed to decode any sonar channel data from XTF pings.")

    if len(chan_arrays) >= 2:
        port_raw = chan_arrays[0]
        stbd_raw = chan_arrays[1]
        # In standard XTF: Port nadir is at the right edge of Port array;
        # Starboard nadir is at the left edge of Starboard array.
        # Align: [Port (left to nadir), Starboard (nadir to right)]
        unified_waterfall = np.hstack([port_raw, stbd_raw])
    else:
        unified_waterfall = chan_arrays[0]

    # Normalize acoustic samples to 8-bit uint8 representation [0, 255]
    # Preserve 99.5th percentile contrast clipping for hydrographic clarity
    wf_float = unified_waterfall.astype(np.float32)
    p_min = float(np.percentile(wf_float, 1))
    p_max = float(np.percentile(wf_float, 99.5))
    if p_max > p_min:
        norm_gray = np.clip((wf_float - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm_gray = np.clip(wf_float, 0, 255).astype(np.uint8)

    # Optional Slant Range Correction (removes water column deadband)
    altitude_px = None
    if apply_slant_correction:
        try:
            norm_gray, altitude_px = correct_slant_range_waterfall(norm_gray)
            is_slant_corrected = True
        except Exception as e:
            logger.warning("[XTF Parser] Slant range correction notice: %s", e)
            is_slant_corrected = False
    else:
        is_slant_corrected = False

    # Create 3-channel BGR image for inference engine
    img_bgr = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)

    sonar_name = getattr(file_header, "SonarName", "Triton XTF Side-Scan Sonar")
    if isinstance(sonar_name, bytes):
        sonar_name = sonar_name.decode("utf-8", errors="ignore").strip()

    return {
        "raw_gray": norm_gray,
        "img_bgr": img_bgr,
        "width": int(norm_gray.shape[1]),
        "height": int(norm_gray.shape[0]),
        "channels": 1,
        "num_pings": num_pings,
        "sonar_channels": num_channels,
        "sonar_name": str(sonar_name),
        "towfish_altitude_m": mean_altitude,
        "towfish_depth_m": mean_depth,
        "heading_deg": mean_heading,
        "pitch_deg": mean_pitch,
        "roll_deg": mean_roll,
        "latitude": lat_val if (lat_val and -90 <= lat_val <= 90) else None,
        "longitude": lon_val if (lon_val and -180 <= lon_val <= 180) else None,
        "slant_range_corrected": is_slant_corrected,
        "detected_altitude_px": altitude_px
    }
