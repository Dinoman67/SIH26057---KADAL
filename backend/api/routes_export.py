import os
import json
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import RESULTS_DIR
from backend.reports.nmea_export import generate_nmea_export
from backend.reports.kml_export import generate_kml_export
from backend.reports.csv_report import generate_csv_report

router = APIRouter(prefix="/export", tags=["Exports"])

VALID_VERDICTS = {"confirmed", "rejected"}


class VerdictUpdate(BaseModel):
    verdicts: Dict[str, str]


def get_analysis_dir(analysis_id: str) -> Path:
    analysis_dir = RESULTS_DIR / analysis_id
    if not analysis_dir.exists():
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    return analysis_dir


def load_verdicts(analysis_id: str) -> Dict[str, str]:
    verdict_file = get_analysis_dir(analysis_id) / "verdicts.json"
    if not verdict_file.exists():
        return {}
    try:
        data = json.loads(verdict_file.read_text())
        return {str(k): v for k, v in data.items()
                if v in VALID_VERDICTS}
    except Exception:
        return {}


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


# -------------------------------------------------------------------------
# Operator Review Verdicts (PATCH/GET /api/export/{analysis_id}/verdicts)
# -------------------------------------------------------------------------

def _refresh_review_artifacts(analysis_id: str, verdicts: Dict[str, str]) -> Dict[str, int]:
    """Persists verdicts, stamps them into results.json + detections.csv."""
    analysis_dir = get_analysis_dir(analysis_id)
    (analysis_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2))

    results_file = analysis_dir / "results.json"
    counts = {"confirmed": 0, "rejected": 0, "pending": 0}
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text())
            dets = data.get("detections", [])
            for det in dets:
                v = verdicts.get(str(det.get("id")))
                det["review_verdict"] = v
                counts["confirmed" if v == "confirmed" else "rejected" if v == "rejected" else "pending"] += 1
            results_file.write_text(json.dumps(data, indent=2, default=str))
            geo_meta = data.get("geospatial_metadata", {})
            (analysis_dir / "detections.csv").write_text(
                generate_csv_report(dets, geo_meta, verdicts))
        except Exception:
            pass
    return counts


@router.get("/{analysis_id}/verdicts")
async def get_review_verdicts(analysis_id: str):
    """Returns stored operator review verdicts for an analysis."""
    return {"analysis_id": analysis_id, "verdicts": load_verdicts(analysis_id)}


@router.patch("/{analysis_id}/verdicts")
async def save_review_verdicts(analysis_id: str, update: VerdictUpdate):
    """
    Stores operator review verdicts ({detection_id: 'confirmed'|'rejected';
    'pending'/null clears}) and stamps them into results.json + detections.csv.
    """
    get_analysis_dir(analysis_id)
    cleaned: Dict[str, str] = {}
    for k, v in (update.verdicts or {}).items():
        if v in VALID_VERDICTS:
            cleaned[str(k)] = v
        # 'pending'/null/unknown values clear the verdict (key dropped)
    counts = _refresh_review_artifacts(analysis_id, cleaned)
    return {"analysis_id": analysis_id, "verdicts": cleaned, "counts": counts}


# -------------------------------------------------------------------------
# Evidence Bundle (GET /api/export/{analysis_id}/bundle)
# -------------------------------------------------------------------------

BUNDLE_FILES = [
    ("annotated.png", "annotated.png"),
    ("evidence.png", "evidence.png"),
    ("colormap.png", "colormap.png"),
    ("detections.csv", "detections.csv"),
    ("results.json", "results.json"),
    ("report.pdf", "report.pdf"),
    ("waypoints.txt", "waypoints.txt"),
    ("dive_plan.kml", "dive_plan.kml"),
]


@router.get("/{analysis_id}/bundle")
async def download_evidence_bundle(analysis_id: str):
    """Downloads a single ZIP evidence bundle (imagery + reports + C2 exports)."""
    analysis_dir = get_analysis_dir(analysis_id)
    ensure_nmea_file(analysis_id)
    ensure_kml_file(analysis_id)
    bundle_path = analysis_dir / "evidence_bundle.zip"
    with zipfile.ZipFile(str(bundle_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, arcname in BUNDLE_FILES:
            src = analysis_dir / filename
            if src.exists():
                zf.write(str(src), arcname)
    return FileResponse(
        path=str(bundle_path),
        media_type="application/zip",
        filename=f"kadal_evidence_{analysis_id[:8]}.zip"
    )
