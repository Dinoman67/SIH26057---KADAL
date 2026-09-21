from fastapi import APIRouter

router = APIRouter(tags=["Waterfall"])


@router.get("/waterfall/surveys")
async def get_waterfall_surveys():
    """Curated demo waterfall transects (simulated; no inference, no storage writes)."""
    return [
        {
            "survey_id": "noaa-mcm-salvage-transect",
            "title": "NOAA Coastal MCM & Salvage Transect (SIMULATED)",
            "strip_image_url": "/static/samples/sample_noaa_debris.png",
            "total_length_meters": 1200.0,
            "swath_width_meters": 150.0,
            "vessel_speed_knots": 4.0,
            "ping_rate_hz": 15.0,
            "targets": [
                {
                    "id": 1,
                    "class_name": "mine",
                    "confidence": 0.884,
                    "y_trigger_px": 420,
                    "bbox": [0.42, 0.21, 0.16, 0.08],
                    "latitude": 29.7142,
                    "longitude": -85.1204,
                    "threat_level": "CRITICAL",
                },
                {
                    "id": 2,
                    "class_name": "wreck",
                    "confidence": 0.792,
                    "y_trigger_px": 1150,
                    "bbox": [0.30, 0.575, 0.40, 0.12],
                    "latitude": 29.7188,
                    "longitude": -85.115,
                    "threat_level": "HIGH",
                },
                {
                    "id": 3,
                    "class_name": "unknown_debris",
                    "confidence": 0.915,
                    "y_trigger_px": 1680,
                    "bbox": [0.55, 0.84, 0.20, 0.07],
                    "latitude": 29.721,
                    "longitude": -85.1112,
                    "threat_level": "MEDIUM",
                },
            ],
        }
    ]
