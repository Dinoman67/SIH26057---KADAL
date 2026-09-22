from typing import Optional, Dict, Any, Tuple
import rasterio.transform
from rasterio.transform import Affine
from pyproj import Transformer
from backend.geospatial.crs import parse_crs

def pixel_to_geographic(
    px: float,
    py: float,
    geo_meta: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Converts pixel coordinates (px=cx, py=cy) to geographic coordinates (lat, lon)
    using the image's genuine geospatial affine transform and CRS.
    Never fabricates coordinates or assigns camera GPS to individual debris detections.
    """
    if not geo_meta.get("georeferenced") or not geo_meta.get("lat_lon_available"):
        return None

    # Only images with a valid Affine transform can resolve pixel-level detection coordinates
    transform = geo_meta.get("transform")
    if transform is None:
        return None

    # Ensure transform is an Affine object
    if not isinstance(transform, Affine):
        try:
            if isinstance(transform, (list, tuple)):
                if len(transform) == 6:
                    transform = Affine(*transform)
                elif len(transform) == 9:
                    transform = Affine(transform[0], transform[1], transform[2],
                                       transform[3], transform[4], transform[5])
        except Exception:
            return None

    raw_crs = geo_meta.get("raw_crs")
    if raw_crs is None and geo_meta.get("crs"):
        raw_crs = parse_crs(geo_meta.get("crs"))

    if raw_crs is None:
        return None

    try:
        # px is continuous column (cx), py is continuous row (cy)
        # transform * (px, py) calculates the exact projected map coordinate
        proj_x, proj_y = transform * (float(px), float(py))

        # Transform from source CRS to WGS84 Lat/Lon (EPSG:4326)
        transformer = Transformer.from_crs(raw_crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(proj_x, proj_y)

        # Check for valid floating point numbers
        if lat is None or lon is None or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None

        is_geographic = False
        try:
            if hasattr(raw_crs, "is_geographic"):
                is_geographic = raw_crs.is_geographic
            elif hasattr(raw_crs, "to_epsg") and raw_crs.to_epsg() in [4326, 4269, 4267]:
                is_geographic = True
        except Exception:
            pass

        return {
            "latitude": round(float(lat), 7),
            "longitude": round(float(lon), 7),
            "crs": geo_meta.get("crs", "EPSG:4326"),
            "coordinate_source": geo_meta.get("coordinate_source", "GeoTIFF Affine Transform"),
            "utm_x": round(float(proj_x), 2) if not is_geographic else None,
            "utm_y": round(float(proj_y), 2) if not is_geographic else None
        }
    except Exception:
        return None

def estimate_position_uncertainty(
    bbox: Dict[str, Any],
    pixel_resolution: Any,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Conservative position search radius for a georeferenced detection:
    half the max box dimension multiplied by pixel resolution.
    Returns (uncertainty_meters, method) or (None, None) when the
    resolution is unknown. Never field-validated — a planning aid, not a claim.
    """
    try:
        if not pixel_resolution:
            return None, None
        rx, ry = float(pixel_resolution[0]), float(pixel_resolution[1])
        if rx <= 0 or ry <= 0:
            return None, None
        w = abs(float(bbox.get("x2", 0)) - float(bbox.get("x1", 0)))
        h = abs(float(bbox.get("y2", 0)) - float(bbox.get("y1", 0)))
        if w <= 0 or h <= 0:
            return None, None
        unc = round(max(rx, ry) * max(w, h) * 0.5, 2)
        method = "half max box-dimension x pixel resolution (conservative search radius; not field-validated)"
        return unc, method
    except Exception:
        return None, None


def compute_crop_transform(
    parent_transform: Affine,
    col_offset: float,
    row_offset: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0
) -> Affine:
    """
    Computes updated affine transform for a crop or resampled sub-window from a parent raster.
    """
    a = parent_transform.a * scale_x
    b = parent_transform.b * scale_y
    c = parent_transform.c + parent_transform.a * col_offset + parent_transform.b * row_offset
    d = parent_transform.d * scale_x
    e = parent_transform.e * scale_y
    f = parent_transform.f + parent_transform.d * col_offset + parent_transform.e * row_offset
    return Affine(a, b, c, d, e, f)

