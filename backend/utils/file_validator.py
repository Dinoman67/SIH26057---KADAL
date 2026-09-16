import os
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import UploadFile, HTTPException

from backend.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB, UPLOADS_DIR

# Georeferencing sidecars accepted alongside an image upload. Matched against
# backend/geospatial/metadata.py::find_sidecar_file (world files, .prj, PAM XML).
ALLOWED_SIDECAR_SUFFIXES = (
    ".tfw", ".tifw", ".wld",
    ".jgw", ".jpgw", ".jpegw",
    ".pgw", ".pngw",
    ".prj", ".aux.xml",
)

def _stream_save(src_file, target_path: Path, max_bytes: int) -> int:
    """Streams src_file to target_path enforcing max_bytes. Returns bytes written."""
    total_bytes = 0
    with open(target_path, "wb") as f:
        while chunk := src_file.read(65536):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                target_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB."
                )
            f.write(chunk)
    return total_bytes

def validate_and_save_upload(
    file: UploadFile,
    unique_id: Optional[str] = None,
) -> Tuple[str, str, int]:
    """
    Validates uploaded file extension, size, and saves to secure temporary storage.
    Returns (saved_file_path, original_filename, file_size_bytes).
    """
    orig_name = file.filename or "uploaded_image"
    ext = Path(orig_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    unique_id = unique_id or uuid.uuid4().hex
    safe_filename = f"{unique_id}_{Path(orig_name).name}"
    target_path = UPLOADS_DIR / safe_filename

    # Stream write and enforce file size limit
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_bytes = _stream_save(file.file, target_path, max_bytes)

    if total_bytes == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return str(target_path), orig_name, total_bytes

def validate_and_save_sidecars(
    sidecars: List[UploadFile],
    saved_image_stem: str,
) -> List[str]:
    """
    Saves georeferencing sidecar files (world files, .prj, PAM XML) next to an
    uploaded image so backend/geospatial/metadata.py::find_sidecar_file can
    discover them. Sidecars take the saved image's stem so stem matching works
    regardless of the sidecar's uploaded filename.
    Returns list of saved sidecar paths (for cleanup by the caller).
    Pass saved_image_stem=Path(saved_image_path).stem of the uploaded image.
    """
    saved: List[str] = []
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for sc in sidecars or []:
        orig_name = sc.filename or ""
        lowered = orig_name.lower()
        matched_suffix = next(
            (sfx for sfx in ALLOWED_SIDECAR_SUFFIXES if lowered.endswith(sfx)),
            None,
        )
        if matched_suffix is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported sidecar '{orig_name}'. Supported: {', '.join(ALLOWED_SIDECAR_SUFFIXES)}"
            )

        target_path = UPLOADS_DIR / f"{saved_image_stem}{matched_suffix}"
        total_bytes = _stream_save(sc.file, target_path, max_bytes)
        if total_bytes == 0:
            target_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Sidecar '{orig_name}' is empty.")
        saved.append(str(target_path))

    return saved
