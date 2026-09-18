import os
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

from backend.config import RESULTS_DIR
from backend.reports.nmea_export import generate_nmea_export
from backend.reports.kml_export import generate_kml_export

router = APIRouter(prefix="/export", tags=["Exports"])


def get_latest_analysis_id() -> str:
    """Finds the most recent analysis directory in RESULTS_DIR."""
    if not RESULTS_DIR.exists():
        raise HTTPException(status_code=404, detail="No analyses found on system.")
    subdirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        raise HTTPException(status_code=404, detail="No analyses found on system.")
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest.name


def ensure_nmea_file(analysis_id: str) -> Path:
    """Retrieves or generates on-the-fly the NMEA 0183 waypoints.txt file."""
    analysis_dir = RESULTS_DIR / analysis_id
    if not analysis_dir.exists():
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    file_path = analysis_dir / "waypoints.txt"
    if not file_path.exists():
        results_file = analysis_dir / "results.json"
        if not results_file.exists():
            raise HTTPException(status_code=404, detail=f"Results data for '{analysis_id}' not found.")
        data = json.loads(results_file.read_text())
        detections = data.get("detections", [])
        src_file = data.get("file_metadata", {}).get("filename", "")
        content = generate_nmea_export(detections, source_filename=src_file, analysis_id=analysis_id)
        file_path.write_text(content)
    return file_path


def ensure_kml_file(analysis_id: str) -> Path:
    """Retrieves or generates on-the-fly the KML dive_plan.kml file."""
    analysis_dir = RESULTS_DIR / analysis_id
    if not analysis_dir.exists():
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    file_path = analysis_dir / "dive_plan.kml"
    if not file_path.exists():
        results_file = analysis_dir / "results.json"
        if not results_file.exists():
            raise HTTPException(status_code=404, detail=f"Results data for '{analysis_id}' not found.")
        data = json.loads(results_file.read_text())
        detections = data.get("detections", [])
        src_file = data.get("file_metadata", {}).get("filename", "")
        content = generate_kml_export(detections, source_filename=src_file, analysis_id=analysis_id)
        file_path.write_text(content)
    return file_path


def get_result_file(analysis_id: str, filename: str, media_type: str):
    analysis_dir = RESULTS_DIR / analysis_id
    file_path = analysis_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Export file '{filename}' for analysis '{analysis_id}' not found.")
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )


# -------------------------------------------------------------------------
# Static C2 Export Endpoints (GET /api/export/nmea and GET /api/export/kml)
# -------------------------------------------------------------------------

@router.get("/nmea")
async def download_nmea_export_direct(analysis_id: Optional[str] = Query(None)):
    """
    Downloads NMEA 0183 $GPWPL waypoint sentences as a downloadable .txt file.
    If analysis_id is omitted, returns the latest analysis.
    """
    target_id = analysis_id or get_latest_analysis_id()
    file_path = ensure_nmea_file(target_id)
    return FileResponse(
        path=str(file_path),
        media_type="text/plain",
        filename="waypoints.txt"
    )


@router.get("/kml")
async def download_kml_export_direct(analysis_id: Optional[str] = Query(None)):
    """
    Downloads Google Earth .kml autonomous dive route plan for direct download.
    If analysis_id is omitted, returns the latest analysis.
    """
    target_id = analysis_id or get_latest_analysis_id()
    file_path = ensure_kml_file(target_id)
    return FileResponse(
        path=str(file_path),
        media_type="application/vnd.google-earth.kml+xml",
        filename="dive_plan.kml"
    )


# -------------------------------------------------------------------------
# Parameterized C2 Export Endpoints (GET /api/export/{analysis_id}/...)
# -------------------------------------------------------------------------

@router.get("/{analysis_id}/nmea")
async def download_nmea_report(analysis_id: str):
    """Downloads NMEA 0183 waypoints for a specific analysis."""
    file_path = ensure_nmea_file(analysis_id)
    return FileResponse(
        path=str(file_path),
        media_type="text/plain",
        filename="waypoints.txt"
    )


@router.get("/{analysis_id}/kml")
async def download_kml_report(analysis_id: str):
    """Downloads Google Earth KML dive route for a specific analysis."""
    file_path = ensure_kml_file(analysis_id)
    return FileResponse(
        path=str(file_path),
        media_type="application/vnd.google-earth.kml+xml",
        filename="dive_plan.kml"
    )


@router.get("/{analysis_id}/pdf")
async def download_pdf_report(analysis_id: str):
    return get_result_file(analysis_id, "report.pdf", "application/pdf")


@router.get("/{analysis_id}/csv")
async def download_csv_report(analysis_id: str):
    return get_result_file(analysis_id, "detections.csv", "text/csv")


@router.get("/{analysis_id}/json")
async def download_json_report(analysis_id: str):
    return get_result_file(analysis_id, "results.json", "application/json")


@router.get("/{analysis_id}/annotated")
async def get_annotated_image(analysis_id: str):
    return get_result_file(analysis_id, "annotated.png", "image/png")


@router.get("/{analysis_id}/original")
async def get_original_image(analysis_id: str):
    return get_result_file(analysis_id, "original.png", "image/png")


@router.get("/{analysis_id}/mask")
async def get_detection_mask(analysis_id: str):
    return get_result_file(analysis_id, "mask.png", "image/png")


@router.get("/{analysis_id}/colormap")
async def get_colormap_image(analysis_id: str):
    return get_result_file(analysis_id, "colormap.png", "image/png")


@router.get("/{analysis_id}/evidence")
async def get_evidence_image(analysis_id: str):
    return get_result_file(analysis_id, "evidence.png", "image/png")
