from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

def format_object_type(class_name: str) -> str:
    """Map raw class names to standardized MoES SIH26057 domain keywords."""
    c = (class_name or "").lower().strip()
    if c in ("unknown_debris", "marine_debris", "debris"):
        return "Entangled Net / Marine Debris"
    elif c in ("mine", "cylinder", "pipe"):
        return "Cylinder / Pipe"
    elif c in ("wreck", "shipwreck"):
        return "Shipwreck"
    elif c in ("airplane", "aircraft"):
        return "Submerged Aircraft"
    return c.replace("_", " ").title()

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X pixel coordinate")
    y1: float = Field(..., description="Top-left Y pixel coordinate")
    x2: float = Field(..., description="Bottom-right X pixel coordinate")
    y2: float = Field(..., description="Bottom-right Y pixel coordinate")

class CenterPixel(BaseModel):
    x: float = Field(..., description="Center X pixel coordinate")
    y: float = Field(..., description="Center Y pixel coordinate")

class Geolocation(BaseModel):
    latitude: float = Field(..., description="WGS84 Latitude in decimal degrees")
    longitude: float = Field(..., description="WGS84 Longitude in decimal degrees")
    crs: str = Field(..., description="Coordinate reference system of source imagery")
    coordinate_source: str = Field(..., description="Method used to derive coordinates (GeoTIFF transform, EXIF GPS)")
    utm_x: Optional[float] = Field(None, description="Projected Easting coordinate if applicable")
    utm_y: Optional[float] = Field(None, description="Projected Northing coordinate if applicable")

class DetectionRecord(BaseModel):
    id: int = Field(..., description="Unique detection sequence ID")
    class_id: int = Field(..., description="Class integer ID")
    class_name: str = Field(..., description="Class name label")
    object_type: Optional[str] = Field(None, description="Human-readable object classification type")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    bbox: BoundingBox = Field(..., description="Detection bounding box in pixel space")
    center_pixel: CenterPixel = Field(..., description="Center point in pixel space")
    geolocation: Optional[Geolocation] = Field(None, description="Geographic coordinates if georeferenced")
    material_density: Optional[str] = Field(None, description="Classified material density: 'Hard (Metallic)' vs 'Soft (Synthetic/Plastic)'")
    estimated_height_meters: Optional[float] = Field(None, description="Hydrographic target mensuration height in meters")
    target_length_meters: Optional[float] = Field(None, description="Physical target footprint length in meters")
    target_width_meters: Optional[float] = Field(None, description="Physical target footprint width in meters")
    peak_backscatter_p95: Optional[float] = Field(None, description="95th percentile acoustic backscatter intensity [0-255]")
    shadow_length_meters: Optional[float] = Field(None, description="Acoustic shadow length in meters")
    threat_score: int = Field(default=0, description="Automated C2 threat score [0-100] based on material density, relief height, and confidence")
    uncertainty_meters: Optional[float] = Field(None, description="Conservative position search radius in meters (half max box-dimension x pixel resolution); None when not georeferenced")
    uncertainty_method: Optional[str] = Field(None, description="Method used to derive uncertainty_meters")
    review_verdict: Optional[str] = Field(None, description="Operator review verdict: 'confirmed', 'rejected', or None (pending)")

class FileMetadata(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    file_size_bytes: int
    file_size_human: str
    channels: int

class GeospatialMetadata(BaseModel):
    georeferenced: bool
    crs: Optional[str] = None
    bounds: Optional[Dict[str, Any]] = None
    pixel_resolution: Optional[List[float]] = None
    lat_lon_available: bool = False
    coordinate_source: Optional[str] = None
    status_message: str
    camera_latitude: Optional[float] = None
    camera_longitude: Optional[float] = None
    camera_altitude: Optional[float] = None
    capture_direction: Optional[float] = None
    footprint_geojson: Optional[Dict[str, Any]] = None
    towfish_altitude_m: Optional[float] = None
    slant_range_corrected: Optional[bool] = None

class ModelMetadata(BaseModel):
    model_name: str = "YOLOv8-ESI"
    format: str = "ONNX"
    input_resolution: str = "256x256"
    execution_provider: str
    model_path: str
    sha256_hash: str
    classes: Dict[int, str]
    architecture: str = "YOLOv8n + Squeeze-and-Excitation (SE) Attention"
    attention_mechanism: str = "SE Channel Attention in C2f Feature Blocks"

class AnalysisSummary(BaseModel):
    debris_detected: bool
    total_detections: int
    highest_confidence: Optional[float] = None
    average_confidence: Optional[float] = None
    class_counts: Dict[str, int] = Field(default_factory=dict)
    material_counts: Dict[str, int] = Field(default_factory=dict, description="Distribution of detected materials (Metallic vs Synthetic)")
    detected_object_types: List[str] = Field(default_factory=list, description="Human-readable object types detected")
    primary_object_type: Optional[str] = Field(None, description="Primary detected object type")
    inference_time_ms: float
    total_time_ms: float
    status: str
    message: str
    noise_filtering_active: bool = Field(default=True, description="Whether adaptive Rayleigh speckle and clutter suppression is active")


class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    summary: AnalysisSummary
    detections: List[DetectionRecord]
    file_metadata: FileMetadata
    geospatial_metadata: GeospatialMetadata
    model_metadata: ModelMetadata
    original_image_url: str
    annotated_image_url: str
    detection_mask_url: str
    colormap_image_url: str
    evidence_image_url: str
    csv_export_url: str
    json_export_url: str
    pdf_report_url: str
    nmea_export_url: Optional[str] = None
    kml_export_url: Optional[str] = None
    bundle_export_url: Optional[str] = None
