"""
backend/reports/nmea_export.py

NMEA 0183 Naval Bridge Export Module for SonarVision C2 Suite.
Generates standard NMEA 0183 $GPWPL (Waypoint Location) sentences with 
XOR checksums for integration with naval ECDIS, radar, and GPS chartplotters.
"""

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


def format_nmea_latitude(lat: float) -> Tuple[str, str]:
    """
    Converts decimal latitude to NMEA 0183 format: DDMM.MMMM
    Returns (lat_str, direction) where direction is 'N' or 'S'.
    """
    direction = "N" if lat >= 0 else "S"
    abs_lat = abs(lat)
    degrees = int(abs_lat)
    minutes = (abs_lat - degrees) * 60.0
    lat_str = f"{degrees:02d}{minutes:07.4f}"
    return lat_str, direction


def format_nmea_longitude(lon: float) -> Tuple[str, str]:
    """
    Converts decimal longitude to NMEA 0183 format: DDDMM.MMMM
    Returns (lon_str, direction) where direction is 'E' or 'W'.
    """
    direction = "E" if lon >= 0 else "W"
    abs_lon = abs(lon)
    degrees = int(abs_lon)
    minutes = (abs_lon - degrees) * 60.0
    lon_str = f"{degrees:03d}{minutes:07.4f}"
    return lon_str, direction


def calculate_nmea_checksum(sentence_body: str) -> str:
    """
    Calculates the standard NMEA 0183 8-bit XOR checksum.
    The checksum is computed across all characters between '$' and '*' (exclusive).
    """
    csum = 0
    for char in sentence_body:
        csum ^= ord(char)
    return f"{csum:02X}"


def build_gpwpl_sentence(lat: float, lon: float, waypoint_id: str) -> str:
    """
    Constructs a standard NMEA 0183 $GPWPL sentence.
    Format: $GPWPL,llll.ll,a,yyyyy.yy,a,c--c*hh<CR><LF>
    """
    lat_str, lat_dir = format_nmea_latitude(lat)
    lon_str, lon_dir = format_nmea_longitude(lon)
    body = f"GPWPL,{lat_str},{lat_dir},{lon_str},{lon_dir},{waypoint_id}"
    checksum = calculate_nmea_checksum(body)
    return f"${body}*{checksum}\r\n"


def generate_nmea_export(
    detections: List[Dict[str, Any]],
    source_filename: Optional[str] = None,
    analysis_id: Optional[str] = None
) -> str:
    """
    Generates a complete NMEA 0183 waypoint transmission stream from a list of detections.
    Each georeferenced contact is converted into a standard $GPWPL waypoint sentence.
    """
    lines = [
        "!--- SONARVISION C2 NAVAL BRIDGE EXPORT ---!",
        f"# System: SonarVision Autonomous Edge Intelligence (SIH26057)",
        f"# Protocol: NMEA 0183 v4.10 ($GPWPL Waypoint Location)",
        f"# Export Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]
    if source_filename:
        lines.append(f"# Source File: {source_filename}")
    if analysis_id:
        lines.append(f"# Analysis ID: {analysis_id}")
    lines.append("# -------------------------------------------------------------")

    georeferenced_count = 0

    for idx, det in enumerate(detections):
        geo = det.get("geolocation")
        # Handle dict or Pydantic model
        if hasattr(geo, "latitude") and hasattr(geo, "longitude"):
            lat = getattr(geo, "latitude")
            lon = getattr(geo, "longitude")
        elif isinstance(geo, dict):
            lat = geo.get("latitude")
            lon = geo.get("longitude")
        else:
            lat = None
            lon = None

        if lat is None or lon is None:
            continue

        georeferenced_count += 1
        det_id = det.get("id", idx + 1)
        waypoint_id = f"WP{det_id:03d}"

        # Extract tactical metadata
        cname = det.get("class_name", "debris")
        obj_type = det.get("object_type", cname.replace("_", " ").title())
        threat = det.get("threat_score", 0)
        material = det.get("material_density", "Unclassified")
        height = det.get("estimated_height_meters")
        h_str = f"{height:.2f}m" if (height is not None and height > 0) else "N/A"
        conf = det.get("confidence", 0.0)

        # Tactical ECDIS comment banner
        lines.append(
            f"# Waypoint: {waypoint_id} | Type: {obj_type} | Threat: {threat}/100 | "
            f"Material: {material} | Relief: {h_str} | Conf: {conf*100:.1f}%"
        )
        # NMEA Sentence
        sentence = build_gpwpl_sentence(float(lat), float(lon), waypoint_id)
        lines.append(sentence.strip())

    if georeferenced_count == 0:
        lines.append("# NOTICE: Zero georeferenced targets found in this sonogram.")
        lines.append("# Check if source file has GeoTIFF affine tags, EXIF GPS, or XTF navigation packets.")

    lines.append("# --- END OF NMEA TRANSMISSION ---")
    return "\r\n".join(lines) + "\r\n"
