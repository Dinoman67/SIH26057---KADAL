"""
tests/test_review_exports.py

Review queue (operator verdicts), evidence bundle, and position
uncertainty — all without touching the model.
"""

import json
import shutil
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.config import RESULTS_DIR
from backend.geospatial.coordinates import estimate_position_uncertainty


ANALYSIS_ID = "test_review_unit_001"


def _seed_analysis():
    d = RESULTS_DIR / ANALYSIS_ID
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    dets = [{
        "id": 1, "class_id": 0, "class_name": "unknown_debris",
        "object_type": "Entangled Net / Marine Debris", "confidence": 0.9,
        "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 60},
        "center_pixel": {"x": 30, "y": 35},
        "geolocation": {"latitude": 28.9, "longitude": -89.4,
                        "crs": "EPSG:26916", "coordinate_source": "GeoTIFF Affine Transform"},
        "material_density": "Soft (Synthetic/Plastic)",
        "uncertainty_meters": 10.0,
        "threat_score": 40, "review_verdict": None,
    }]
    (d / "results.json").write_text(json.dumps({
        "analysis_id": ANALYSIS_ID,
        "detections": dets,
        "geospatial_metadata": {"georeferenced": True, "crs": "EPSG:26916",
                                "coordinate_source": "GeoTIFF Affine Transform"},
    }))
    (d / "annotated.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (d / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    (d / "detections.csv").write_text("detection_id,review_verdict\n1,pending\n")
    return d


class TestPositionUncertainty(unittest.TestCase):

    def test_uncertainty_from_resolution(self):
        """Half max box-dimension x pixel resolution."""
        unc, method = estimate_position_uncertainty(
            {"x1": 0, "y1": 0, "x2": 40, "y2": 10}, [0.5, 0.5])
        self.assertEqual(unc, 10.0)  # 0.5 * 40 * 0.5
        self.assertIn("search radius", method)

    def test_uncertainty_missing_resolution(self):
        """No resolution -> honest None, never fabricated."""
        unc, method = estimate_position_uncertainty(
            {"x1": 0, "y1": 0, "x2": 40, "y2": 10}, None)
        self.assertIsNone(unc)
        self.assertIsNone(method)


class TestReviewVerdicts(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        _seed_analysis()

    def tearDown(self):
        d = RESULTS_DIR / ANALYSIS_ID
        if d.exists():
            shutil.rmtree(d)

    def test_patch_and_get_verdicts(self):
        """PATCH stores verdicts; GET returns them; CSV is stamped."""
        res = self.client.patch(
            f"/api/export/{ANALYSIS_ID}/verdicts",
            json={"verdicts": {"1": "confirmed", "2": "bogus", "3": "pending"}})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["verdicts"], {"1": "confirmed"})
        self.assertEqual(body["counts"]["confirmed"], 1)

        res = self.client.get(f"/api/export/{ANALYSIS_ID}/verdicts")
        self.assertEqual(res.json()["verdicts"], {"1": "confirmed"})

        csv_text = (RESULTS_DIR / ANALYSIS_ID / "detections.csv").read_text()
        self.assertIn("review_verdict", csv_text.splitlines()[0])
        self.assertIn("confirmed", csv_text)

    def test_verdicts_404(self):
        """Unknown analysis id -> 404, not 500."""
        res = self.client.patch("/api/export/nope/verdicts",
                                json={"verdicts": {"1": "confirmed"}})
        self.assertEqual(res.status_code, 404)


class TestEvidenceBundle(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        _seed_analysis()

    def tearDown(self):
        d = RESULTS_DIR / ANALYSIS_ID
        if d.exists():
            shutil.rmtree(d)

    def test_bundle_zip_contents(self):
        """Bundle ZIP contains imagery + stamped reports."""
        res = self.client.get(f"/api/export/{ANALYSIS_ID}/bundle")
        self.assertEqual(res.status_code, 200)
        self.assertIn("zip", res.headers.get("content-type", ""))
        tmp = Path("/tmp") / f"{ANALYSIS_ID}.zip"
        tmp.write_bytes(res.content)
        with zipfile.ZipFile(str(tmp)) as zf:
            names = set(zf.namelist())
        tmp.unlink()
        for expected in ("annotated.png", "detections.csv",
                         "results.json", "report.pdf"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
