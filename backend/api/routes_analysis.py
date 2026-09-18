import os
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
import rasterio
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from backend.config import (
    UPLOADS_DIR,
    RESULTS_DIR,
    SAMPLES_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)
from backend.inference.engine import YOLOESIInferenceEngine
from backend.inference.xtf_parser import parse_xtf_file
from backend.geospatial.metadata import extract_geospatial_metadata
from backend.geospatial.coordinates import pixel_to_geographic
from backend.utils.annotator import draw_annotations, generate_detection_only_view, apply_pseudo_colormap, generate_evidence_panel
from backend.utils.file_validator import validate_and_save_upload, validate_and_save_sidecars
from backend.reports.pdf_report import create_pdf_report
from backend.reports.csv_report import generate_csv_report
from backend.reports.json_report import generate_json_report
from backend.reports.nmea_export import generate_nmea_export
from backend.reports.kml_export import generate_kml_export
from backend.schemas.detection import (
    AnalysisResponse, AnalysisSummary, DetectionRecord, FileMetadata,
    GeospatialMetadata, ModelMetadata, BoundingBox, CenterPixel, Geolocation
)

router = APIRouter(tags=["Analysis"])

def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def load_image_to_numpy(image_path: str) -> Tuple[np.ndarray, np.ndarray, int, int, int, Optional[Dict[str, Any]]]:
    """
    Safely loads TIFF, GeoTIFF, XTF, JPG, PNG into uint8 BGR numpy array and preserves
    the un-letterboxed, raw acoustic grayscale numpy array for acoustic physics backscatter analysis.
    Returns: (img_bgr, raw_gray, width, height, channels, xtf_telemetry)
    """
    path = str(image_path)
    ext = Path(path).suffix.lower()

    # Native eXtended Triton Format (.XTF) ingestion
    if ext == ".xtf":
        try:
            xtf_data = parse_xtf_file(path, apply_slant_correction=True)
            return (
                xtf_data["img_bgr"],
                xtf_data["raw_gray"],
                xtf_data["width"],
                xtf_data["height"],
                xtf_data["channels"],
                xtf_data
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to ingest raw .XTF sonar file: {e}")

    if ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(path) as src:
                # Read first 3 bands or single band
                if src.count >= 3:
                    arr = src.read([1, 2, 3])
                    arr = np.transpose(arr, (1, 2, 0)) # H, W, 3
                    raw_gray = np.dot(arr[..., :3], [0.114, 0.587, 0.299]).astype(np.float32)
                else:
                    arr = src.read(1) # H, W
                    raw_gray = arr.astype(np.float32)
                
                # Normalize for BGR visualization & neural inference
                if arr.dtype != np.uint8:
                    min_v, max_v = arr.min(), arr.max()
                    if max_v > min_v:
                        arr_norm = ((arr - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
                    else:
                        arr_norm = arr.astype(np.uint8)
                else:
                    arr_norm = arr.copy()

                if len(arr_norm.shape) == 2:
                    bgr = cv2.cvtColor(arr_norm, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = cv2.cvtColor(arr_norm, cv2.COLOR_RGB2BGR)
                    
                h, w = bgr.shape[:2]
                channels = 3
                return bgr, raw_gray, w, h, channels, None
        except Exception:
            pass

    # Standard fallback via OpenCV / PIL
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            pil_img = Image.open(path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image file: {e}")

    if len(img.shape) == 2:
        h, w = img.shape
        channels = 1
        raw_gray = img.copy().astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        h, w, channels = img.shape
        if channels == 4:
            raw_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            channels = 3
        else:
            raw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    return img, raw_gray, w, h, channels, None

def format_object_type(class_name: str) -> str:
    """Maps internal model class name to a clean human-readable object type."""
    mapping = {
        "unknown_debris": "Marine Debris",
        "marine_debris": "Marine Debris",
        "airplane": "Submerged Aircraft",
        "mine": "Naval Mine",
        "wreck": "Shipwreck",
    }
    return mapping.get(class_name.lower(), class_name.replace("_", " ").title())

def run_full_pipeline(
    image_path: str,
    orig_filename: str,
    file_size_bytes: int,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    use_tiling: Optional[bool] = None
) -> AnalysisResponse:
    t_start = time.perf_counter()
    analysis_id = uuid.uuid4().hex
    analysis_dir = RESULTS_DIR / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image and preserve raw un-stretched acoustic grayscale array
    img_bgr, raw_gray, width, height, channels, xtf_telemetry = load_image_to_numpy(image_path)
    file_format = Path(orig_filename).suffix.upper().replace(".", "")

    # 2. Extract Geospatial Metadata
    geo_meta_raw = extract_geospatial_metadata(image_path, orig_filename=orig_filename)

    # Merge XTF navigation & sonar telemetry if available
    if xtf_telemetry:
        if xtf_telemetry.get("latitude") is not None and xtf_telemetry.get("longitude") is not None:
            geo_meta_raw["georeferenced"] = True
            geo_meta_raw["lat_lon_available"] = True
            geo_meta_raw["coordinate_source"] = "XTF Navigation Telemetry"
            geo_meta_raw["camera_latitude"] = xtf_telemetry.get("latitude")
            geo_meta_raw["camera_longitude"] = xtf_telemetry.get("longitude")
            geo_meta_raw["camera_altitude"] = xtf_telemetry.get("towfish_depth_m")
            geo_meta_raw["capture_direction"] = xtf_telemetry.get("heading_deg")
        geo_meta_raw["towfish_altitude_m"] = xtf_telemetry.get("towfish_altitude_m")
        geo_meta_raw["slant_range_corrected"] = xtf_telemetry.get("slant_range_corrected")
        geo_meta_raw["status_message"] = (
            f"XTF Swath Ingestion: {xtf_telemetry.get('num_pings', 0)} pings | "
            f"Towfish Alt: {xtf_telemetry.get('towfish_altitude_m', 0.0):.1f}m | "
            f"Slant Range Corrected: {xtf_telemetry.get('slant_range_corrected', False)}"
        )

    towfish_alt = (xtf_telemetry.get("towfish_altitude_m") if xtf_telemetry else None) or geo_meta_raw.get("camera_altitude")
    pixel_res = geo_meta_raw.get("pixel_resolution")

    # 3. Run YOLO-ESI ONNX inference with Acoustic Physics & Mensuration
    engine = YOLOESIInferenceEngine()
    detections_raw, timing = engine.predict(
        img_bgr,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        use_tiling=use_tiling,
        raw_gray=raw_gray,
        towfish_altitude_m=towfish_alt,
        pixel_resolution=pixel_res
    )

    # 4. Resolve geographic coordinates and populate acoustic physics telemetry
    detections_list: List[DetectionRecord] = []
    class_counts: Dict[str, int] = {}
    material_counts: Dict[str, int] = {}
    confidences = []

    for d in detections_raw:
        cid = d["class_id"]
        cname = d["class_name"]
        conf = d["confidence"]
        confidences.append(conf)
        class_counts[cname] = class_counts.get(cname, 0) + 1
        obj_type = format_object_type(cname)

        mat_density = d.get("material_density", "Unclassified Benthic Target")
        material_counts[mat_density] = material_counts.get(mat_density, 0) + 1

        # Calculate geospatial coordinates using image metadata
        geo_dict = pixel_to_geographic(
            d["center_pixel"]["x"],
            d["center_pixel"]["y"],
            geo_meta_raw
        )
        geolocation = Geolocation(**geo_dict) if geo_dict else None

        detections_list.append(DetectionRecord(
            id=d["id"],
            class_id=cid,
            class_name=cname,
            object_type=obj_type,
            confidence=conf,
            bbox=BoundingBox(**d["bbox"]),
            center_pixel=CenterPixel(**d["center_pixel"]),
            geolocation=geolocation,
            material_density=d.get("material_density"),
            estimated_height_meters=d.get("estimated_height_meters"),
            peak_backscatter_p95=d.get("peak_backscatter_p95"),
            shadow_length_meters=d.get("shadow_length_meters"),
            threat_score=d.get("threat_score", 0)
        ))

    # 5. Generate Visualizations
    annotated_bgr = draw_annotations(img_bgr, detections_raw)
    mask_bgr = generate_detection_only_view(img_bgr, detections_raw)

    # New: Pseudo-color colormap — highlights debris regions in vibrant color
    colormap_bgr = apply_pseudo_colormap(img_bgr, detections_raw)

    # New: Evidence panel — proves model distinguishes debris from mere brightness
    evidence_bgr = generate_evidence_panel(img_bgr, detections_raw)

    # Save artifacts
    orig_save_path = analysis_dir / "original.png"
    annotated_save_path = analysis_dir / "annotated.png"
    mask_save_path = analysis_dir / "mask.png"
    colormap_save_path = analysis_dir / "colormap.png"
    evidence_save_path = analysis_dir / "evidence.png"
    pdf_save_path = analysis_dir / "report.pdf"
    csv_save_path = analysis_dir / "detections.csv"
    json_save_path = analysis_dir / "results.json"
    nmea_save_path = analysis_dir / "waypoints.txt"
    kml_save_path = analysis_dir / "dive_plan.kml"

    cv2.imwrite(str(orig_save_path), img_bgr)
    cv2.imwrite(str(annotated_save_path), annotated_bgr)
    cv2.imwrite(str(mask_save_path), mask_bgr)
    cv2.imwrite(str(colormap_save_path), colormap_bgr)
    cv2.imwrite(str(evidence_save_path), evidence_bgr)

    # 6. Build Summary Metrics
    total_dets = len(detections_list)
    debris_detected = total_dets > 0
    max_conf = max(confidences) if confidences else None
    avg_conf = (sum(confidences) / len(confidences)) if confidences else None

    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    detected_types = list(dict.fromkeys([format_object_type(c) for c in class_counts.keys()]))
    primary_type = format_object_type(detections_list[0].class_name) if detections_list else None

    # Material classification breakdown summary string
    mat_summary_parts = [f"{m}: {cnt}" for m, cnt in material_counts.items()]
    mat_summary_str = f" [{', '.join(mat_summary_parts)}]" if mat_summary_parts else ""

    summary = AnalysisSummary(
        debris_detected=debris_detected,
        total_detections=total_dets,
        highest_confidence=round(max_conf, 4) if max_conf else None,
        average_confidence=round(avg_conf, 4) if avg_conf else None,
        class_counts=class_counts,
        material_counts=material_counts,
        detected_object_types=detected_types,
        primary_object_type=primary_type,
        inference_time_ms=timing["inference_time_ms"],
        total_time_ms=total_time_ms,
        status="SUCCESS",
        message=f"Detected {total_dets} target(s): {', '.join(detected_types)}{mat_summary_str}." if total_dets > 0 else f"No targets above {conf_threshold:.0%} confidence — raise no alarm, or lower the threshold and re-analyze."
    )

    file_metadata = FileMetadata(
        filename=orig_filename,
        format=file_format,
        width=width,
        height=height,
        file_size_bytes=file_size_bytes,
        file_size_human=format_file_size(file_size_bytes),
        channels=channels
    )

    geospatial_metadata = GeospatialMetadata(
        georeferenced=geo_meta_raw["georeferenced"],
        crs=geo_meta_raw.get("crs"),
        bounds=geo_meta_raw.get("bounds"),
        pixel_resolution=geo_meta_raw.get("pixel_resolution"),
        lat_lon_available=geo_meta_raw.get("lat_lon_available", False),
        coordinate_source=geo_meta_raw.get("coordinate_source"),
        status_message=geo_meta_raw.get("status_message"),
        camera_latitude=geo_meta_raw.get("camera_latitude"),
        camera_longitude=geo_meta_raw.get("camera_longitude"),
        camera_altitude=geo_meta_raw.get("camera_altitude"),
        capture_direction=geo_meta_raw.get("capture_direction"),
        footprint_geojson=geo_meta_raw.get("footprint_geojson"),
        towfish_altitude_m=geo_meta_raw.get("towfish_altitude_m"),
        slant_range_corrected=geo_meta_raw.get("slant_range_corrected")
    )

    model_metadata = ModelMetadata(
        **engine.get_metadata()
    )

    # Construct response dictionary for exports
    analysis_dict = {
        "analysis_id": analysis_id,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary.model_dump(),
        "detections": [d.model_dump() for d in detections_list],
        "file_metadata": file_metadata.model_dump(),
        "geospatial_metadata": geospatial_metadata.model_dump(),
        "model_metadata": model_metadata.model_dump()
    }

    # 7. Generate Reports & C2 Naval Exports
    csv_content = generate_csv_report([d.model_dump() for d in detections_list], geo_meta_raw)
    csv_save_path.write_text(csv_content)

    json_content = generate_json_report(analysis_dict)
    json_save_path.write_text(json_content)

    nmea_content = generate_nmea_export([d.model_dump() for d in detections_list], orig_filename, analysis_id)
    nmea_save_path.write_text(nmea_content)

    kml_content = generate_kml_export([d.model_dump() for d in detections_list], orig_filename, analysis_id)
    kml_save_path.write_text(kml_content)

    create_pdf_report(
        analysis_data=analysis_dict,
        annotated_image_path=str(annotated_save_path),
        output_path=str(pdf_save_path)
    )

    return AnalysisResponse(
        analysis_id=analysis_id,
        timestamp=analysis_dict["timestamp"],
        summary=summary,
        detections=detections_list,
        file_metadata=file_metadata,
        geospatial_metadata=geospatial_metadata,
        model_metadata=model_metadata,
        original_image_url=f"/api/export/{analysis_id}/original",
        annotated_image_url=f"/api/export/{analysis_id}/annotated",
        detection_mask_url=f"/api/export/{analysis_id}/mask",
        colormap_image_url=f"/api/export/{analysis_id}/colormap",
        evidence_image_url=f"/api/export/{analysis_id}/evidence",
        csv_export_url=f"/api/export/{analysis_id}/csv",
        json_export_url=f"/api/export/{analysis_id}/json",
        pdf_report_url=f"/api/export/{analysis_id}/pdf",
        nmea_export_url=f"/api/export/{analysis_id}/nmea",
        kml_export_url=f"/api/export/{analysis_id}/kml"
    )

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    sidecars: Optional[List[UploadFile]] = File(None),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD),
    iou_threshold: float = Form(DEFAULT_IOU_THRESHOLD),
    use_tiling: Optional[bool] = Form(None)
):
    """
    Analyzes an uploaded image. Optionally accepts georeferencing sidecar
    files (world files .tfw/.jgw/.pgw/.wld, .prj, .aux.xml) under the
    'sidecars' form field so plain PNG/JPG uploads resolve to real-world
    coordinates. GeoTIFF-embedded transforms and EXIF GPS need no sidecars.
    """
    saved_path, orig_name, file_size = validate_and_save_upload(file)
    saved_sidecars: List[str] = []
    try:
        if sidecars:
            saved_sidecars = validate_and_save_sidecars(
                sidecars, Path(saved_path).stem
            )
        response = run_full_pipeline(
            image_path=saved_path,
            orig_filename=orig_name,
            file_size_bytes=file_size,
            conf_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            use_tiling=use_tiling
        )
        return response
    finally:
        # Cleanup uploaded raw temp file + sidecars
        Path(saved_path).unlink(missing_ok=True)
        for sc_path in saved_sidecars:
            Path(sc_path).unlink(missing_ok=True)

class SampleRequest(BaseModel):
    sample_id: str
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    use_tiling: Optional[bool] = None

@router.post("/analyze-sample", response_model=AnalysisResponse)
async def analyze_sample_image(req: SampleRequest):
    sample_map = {
        "geotiff_debris": "sample_noaa_geotiff_debris.tif",
        "sss_marine_debris": "sample_sss_marine_debris.png",
        "milco_mine": "sample_milco_mine.png",
        "kaggle_wreck": "sample_kaggle_wreck.png",
        "kaggle_airplane": "sample_kaggle_airplane.png",
        "seabed_background": "sample_seabed_background.png",
        "drone_aerial_geotagged": "sample_drone_aerial_geotagged.jpg"
    }

    if req.sample_id not in sample_map:
        raise HTTPException(status_code=404, detail=f"Unknown sample ID '{req.sample_id}'")

    sample_filename = sample_map[req.sample_id]
    sample_path = SAMPLES_DIR / sample_filename

    if not sample_path.exists():
        from backend.utils.samples_generator import ensure_sample_assets
        ensure_sample_assets()

    if not sample_path.exists():
        raise HTTPException(status_code=500, detail="Sample asset not found on server.")

    file_size = sample_path.stat().st_size
    return run_full_pipeline(
        image_path=str(sample_path),
        orig_filename=sample_filename,
        file_size_bytes=file_size,
        conf_threshold=req.confidence_threshold,
        iou_threshold=req.iou_threshold,
        use_tiling=req.use_tiling
    )
