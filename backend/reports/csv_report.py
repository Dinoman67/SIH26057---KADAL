import csv
import io
from typing import List, Dict, Any

def generate_csv_report(
    detections: List[Dict[str, Any]],
    geospatial_meta: Dict[str, Any],
    verdicts: Dict[Any, str] = None
) -> str:
    """
    Generates structured CSV export for all detections.
    Optional verdicts map ({detection_id: 'confirmed'|'rejected'}) stamps
    operator review decisions; per-detection review_verdict fields win.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "detection_id",
        "threat_score",
        "object_type",
        "class_name",
        "class_id",
        "confidence",
        "material_density",
        "peak_backscatter_p95",
        "estimated_height_meters",
        "target_length_meters",
        "target_width_meters",
        "shadow_length_meters",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "latitude",
        "longitude",
        "crs",
        "coordinate_source",
        "uncertainty_meters",
        "review_verdict"
    ])

    mapping = {
        "unknown_debris": "Entangled Net / Marine Debris",
        "marine_debris": "Entangled Net / Marine Debris",
        "airplane": "Submerged Aircraft",
        "mine": "Cylinder / Pipe",
        "wreck": "Shipwreck",
    }

    for det in detections:
        box = det.get("bbox", {})
        cp = det.get("center_pixel", {})
        geo = det.get("geolocation") or {}
        cname = det.get("class_name", "")
        obj_type = det.get("object_type") or mapping.get(str(cname).lower(), str(cname).replace("_", " ").title())
        det_id = det.get("id")
        verdict = det.get("review_verdict")
        if verdict is None and verdicts:
            verdict = verdicts.get(det_id, verdicts.get(str(det_id)))

        writer.writerow([
            det_id,
            det.get("threat_score", 0),
            obj_type,
            det.get("class_name"),
            det.get("class_id"),
            f"{det.get('confidence', 0.0):.4f}",
            det.get("material_density", ""),
            f"{det.get('peak_backscatter_p95', 0.0):.1f}" if det.get("peak_backscatter_p95") is not None else "",
            f"{det.get('estimated_height_meters', 0.0):.2f}" if det.get("estimated_height_meters") is not None else "",
            f"{det.get('target_length_meters', 0.0):.2f}" if det.get("target_length_meters") is not None else "",
            f"{det.get('target_width_meters', 0.0):.2f}" if det.get("target_width_meters") is not None else "",
            f"{det.get('shadow_length_meters', 0.0):.2f}" if det.get("shadow_length_meters") is not None else "",
            box.get("x1"),
            box.get("y1"),
            box.get("x2"),
            box.get("y2"),
            cp.get("x"),
            cp.get("y"),
            geo.get("latitude") if geo.get("latitude") is not None else "",
            geo.get("longitude") if geo.get("longitude") is not None else "",
            geo.get("crs") if geo.get("crs") is not None else (geospatial_meta.get("crs") if geospatial_meta.get("georeferenced") else ""),
            geo.get("coordinate_source") if geo.get("coordinate_source") is not None else (geospatial_meta.get("coordinate_source") if geospatial_meta.get("georeferenced") else ""),
            f"{det.get('uncertainty_meters', 0.0):.2f}" if det.get("uncertainty_meters") is not None else "",
            verdict or "pending"
        ])

    return output.getvalue()
