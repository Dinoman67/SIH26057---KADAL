"""
tests/test_c2_exports.py

Comprehensive test suite for SonarVision C2 Operational Upgrades:
1. Automated Threat Triage & Risk Score Formula
2. NMEA 0183 Naval Bridge Export & XOR Checksum
3. Autonomous Dive Route Export (.KML) & Nearest-Neighbor Trajectory
4. FastAPI Endpoints (/api/export/nmea and /api/export/kml)
"""

import unittest
from fastapi.testclient import TestClient
from backend.main import app
from backend.inference.acoustic_physics import calculate_threat_score
from backend.reports.nmea_export import (
    format_nmea_latitude,
    format_nmea_longitude,
    calculate_nmea_checksum,
    build_gpwpl_sentence,
    generate_nmea_export
)
from backend.reports.kml_export import (
    haversine_distance,
    sort_points_nearest_neighbor,
    generate_kml_export
)


class TestC2ThreatTriage(unittest.TestCase):

    def test_threat_score_metallic_high_relief(self):
        """High-density metal with 2.0m relief and 90% confidence produces Critical threat (>75)."""
        # Multipliers: Metal = 3.0, Height = 2.0, Conf = 0.90 -> 3.0 * 2.0 * 0.90 * 33.3 = 179.8 -> clamped 100
        score = calculate_threat_score(
            material_density="Hard (Metallic)",
            estimated_height_meters=2.0,
            confidence=0.90
        )
        self.assertEqual(score, 100)

    def test_threat_score_metallic_low_relief(self):
        """High-density metal with 0.8m relief and 80% confidence."""
        # 3.0 * 0.8 * 0.80 * 33.3 = 63.936 -> 64 (Elevated)
        score = calculate_threat_score(
            material_density="Hard (Metallic)",
            estimated_height_meters=0.8,
            confidence=0.80
        )
        self.assertEqual(score, 64)

    def test_threat_score_synthetic_plastic(self):
        """Low-density synthetic with 1.0m relief and 75% confidence produces Monitor threat (<40)."""
        # Multipliers: Synthetic = 1.0, Height = 1.0, Conf = 0.75 -> 1.0 * 1.0 * 0.75 * 33.3 = 24.975 -> 25
        score = calculate_threat_score(
            material_density="Soft (Synthetic/Plastic)",
            estimated_height_meters=1.0,
            confidence=0.75
        )
        self.assertEqual(score, 25)

    def test_threat_score_null_height_default(self):
        """Null or zero estimated height defaults to 1.0 multiplier."""
        # Multipliers: Medium = 2.0, Height = 1.0 (default), Conf = 0.60 -> 2.0 * 1.0 * 0.60 * 33.3 = 39.96 -> 40
        score_none = calculate_threat_score(
            material_density="Unclassified Benthic Target",
            estimated_height_meters=None,
            confidence=0.60
        )
        score_zero = calculate_threat_score(
            material_density="Unclassified Benthic Target",
            estimated_height_meters=0.0,
            confidence=0.60
        )
        self.assertEqual(score_none, 40)
        self.assertEqual(score_zero, 40)

    def test_threat_score_height_capping(self):
        """Height multiplier is capped at 5.0m max."""
        # Height = 15.0m -> capped to 5.0m
        # Synthetic = 1.0, Height = 5.0, Conf = 0.50 -> 1.0 * 5.0 * 0.50 * 33.3 = 83.25 -> 83
        score = calculate_threat_score(
            material_density="Soft (Synthetic/Plastic)",
            estimated_height_meters=15.0,
            confidence=0.50
        )
        self.assertEqual(score, 83)


class TestNMEA0183Export(unittest.TestCase):

    def test_nmea_coordinate_formatting(self):
        """Verify NMEA 0183 DDMM.MMMM and DDDMM.MMMM conversions."""
        # Lat: 28.9166026 N -> 28 deg, 54.9962 min
        lat_str, lat_dir = format_nmea_latitude(28.9166026)
        self.assertEqual(lat_dir, "N")
        self.assertEqual(lat_str, "2854.9962")

        # South lat
        lat_str_s, lat_dir_s = format_nmea_latitude(-12.3456)
        self.assertEqual(lat_dir_s, "S")
        self.assertEqual(lat_str_s, "1220.7360")

        # Lon: -89.4257008 W -> 89 deg, 25.5420 min -> 08925.5420
        lon_str, lon_dir = format_nmea_longitude(-89.4257008)
        self.assertEqual(lon_dir, "W")
        self.assertEqual(lon_str, "08925.5420")

        # East lon
        lon_str_e, lon_dir_e = format_nmea_longitude(72.8777)
        self.assertEqual(lon_dir_e, "E")
        self.assertEqual(lon_str_e, "07252.6620")

    def test_nmea_xor_checksum(self):
        """Verify standard 8-bit XOR checksum calculation."""
        # Sample body: GPWPL,4917.1600,N,12310.6400,W,003
        body = "GPWPL,4917.1600,N,12310.6400,W,003"
        csum = calculate_nmea_checksum(body)
        # Compute expected XOR
        expected = 0
        for c in body:
            expected ^= ord(c)
        self.assertEqual(csum, f"{expected:02X}")

    def test_build_gpwpl_sentence(self):
        """Verify complete $GPWPL sentence structure."""
        sentence = build_gpwpl_sentence(28.9166026, -89.4257008, "WP001")
        self.assertTrue(sentence.startswith("$GPWPL,2854.9962,N,08925.5420,W,WP001*"))
        self.assertTrue(sentence.endswith("\r\n"))

        # Verify sentence checksum validates correctly
        star_idx = sentence.find("*")
        body = sentence[1:star_idx]
        actual_csum = sentence[star_idx+1:].strip()
        self.assertEqual(calculate_nmea_checksum(body), actual_csum)

    def test_generate_nmea_stream(self):
        """Verify full NMEA waypoint transmission generation."""
        mock_dets = [
            {
                "id": 1,
                "class_name": "mine",
                "object_type": "Naval Mine",
                "confidence": 0.89,
                "threat_score": 92,
                "material_density": "Hard (Metallic)",
                "estimated_height_meters": 1.4,
                "geolocation": {"latitude": 28.9166, "longitude": -89.4257}
            },
            {
                "id": 2,
                "class_name": "wreck",
                "object_type": "Shipwreck",
                "confidence": 0.95,
                "threat_score": 85,
                "material_density": "Hard (Metallic)",
                "estimated_height_meters": 3.2,
                "geolocation": {"latitude": 28.9180, "longitude": -89.4270}
            }
        ]
        stream = generate_nmea_export(mock_dets, source_filename="test_sonogram.tif")
        self.assertIn("$GPWPL", stream)
        self.assertIn("WP001", stream)
        self.assertIn("WP002", stream)
        self.assertIn("Naval Mine", stream)
        self.assertIn("Threat: 92/100", stream)


class TestKMLExport(unittest.TestCase):

    def test_nearest_neighbor_ordering(self):
        """Verify nearest neighbor starts with highest threat and connects closest points."""
        points = [
            {"id": 1, "latitude": 28.000, "longitude": -89.000, "threat_score": 30},
            {"id": 2, "latitude": 28.001, "longitude": -89.001, "threat_score": 95}, # Closest to 3
            {"id": 3, "latitude": 28.002, "longitude": -89.002, "threat_score": 50},
        ]
        ordered = sort_points_nearest_neighbor(points)
        # Must start with id 2 (highest threat = 95)
        self.assertEqual(ordered[0]["id"], 2)
        # Next closest to (28.001, -89.001) is (28.002, -89.002) = id 3
        self.assertEqual(ordered[1]["id"], 3)
        # Finally id 1
        self.assertEqual(ordered[2]["id"], 1)

    def test_kml_document_structure(self):
        """Verify valid KML XML output with Placemarks and LineString."""
        mock_dets = [
            {
                "id": 1,
                "class_name": "mine",
                "object_type": "Naval Mine",
                "confidence": 0.90,
                "threat_score": 88,
                "material_density": "Hard (Metallic)",
                "estimated_height_meters": 1.2,
                "peak_backscatter_p95": 210.0,
                "geolocation": {"latitude": 28.9166, "longitude": -89.4257}
            },
            {
                "id": 2,
                "class_name": "unknown_debris",
                "object_type": "Marine Debris",
                "confidence": 0.72,
                "threat_score": 35,
                "material_density": "Soft (Synthetic/Plastic)",
                "estimated_height_meters": 0.5,
                "peak_backscatter_p95": 115.0,
                "geolocation": {"latitude": 28.9170, "longitude": -89.4260}
            }
        ]
        kml = generate_kml_export(mock_dets, source_filename="mission_test.png")
        self.assertIn("<?xml", kml)
        self.assertIn("<kml xmlns=", kml)
        self.assertIn("#criticalThreat", kml)
        self.assertIn("#monitorThreat", kml)
        self.assertIn("<LineString>", kml)
        self.assertIn("AUV Tactical Dive Inspection Route", kml)
        self.assertIn("-89.4257000,28.9166000,0", kml)


class TestExportEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_export_nmea_and_kml_endpoints(self):
        """Verify GET /api/export/nmea and GET /api/export/kml respond properly."""
        # NMEA export direct
        res_nmea = self.client.get("/api/export/nmea")
        self.assertIn(res_nmea.status_code, [200, 404])
        if res_nmea.status_code == 200:
            self.assertIn("GPWPL", res_nmea.text)
            self.assertIn("attachment", res_nmea.headers.get("content-disposition", ""))

        # KML export direct
        res_kml = self.client.get("/api/export/kml")
        self.assertIn(res_kml.status_code, [200, 404])
        if res_kml.status_code == 200:
            self.assertIn("<kml", res_kml.text)
            self.assertIn("attachment", res_kml.headers.get("content-disposition", ""))


if __name__ == "__main__":
    unittest.main()
