import unittest
import numpy as np
from backend.inference.acoustic_physics import (
    classify_material_density,
    calculate_shadow_mensuration,
    soft_nms,
    MATERIAL_HARD_METAL,
    MATERIAL_SOFT_SYNTHETIC
)
from backend.inference.slant_range import (
    detect_first_bottom_return,
    correct_slant_range_scanline,
    correct_slant_range_waterfall
)

class TestAcousticPhysics(unittest.TestCase):

    def test_metallic_backscatter_classification(self):
        """Verify high P95 backscatter classifies as Hard (Metallic)."""
        img = np.full((100, 100), 40, dtype=np.uint8)
        # Add high specular peak in box
        img[20:50, 20:50] = 230
        bbox = {"x1": 20, "y1": 20, "x2": 50, "y2": 50}
        res = classify_material_density(img, bbox)
        self.assertEqual(res["material_density"], MATERIAL_HARD_METAL)
        self.assertGreaterEqual(res["peak_backscatter_p95"], 185.0)
        self.assertGreater(res["impedance_estimate_mrayl"], 30.0)

    def test_synthetic_plastic_classification(self):
        """Verify low P95 backscatter classifies as Soft (Synthetic/Plastic)."""
        img = np.full((100, 100), 40, dtype=np.uint8)
        # Add diffuse return in box
        img[20:50, 20:50] = 110
        bbox = {"x1": 20, "y1": 20, "x2": 50, "y2": 50}
        res = classify_material_density(img, bbox)
        self.assertEqual(res["material_density"], MATERIAL_SOFT_SYNTHETIC)
        self.assertLess(res["peak_backscatter_p95"], 185.0)
        self.assertLess(res["impedance_estimate_mrayl"], 10.0)

    def test_target_mensuration_calculation(self):
        """Verify shadow-based target height mensuration."""
        img = np.full((100, 100), 50, dtype=np.uint8)
        # Highlight
        img[20:40, 20:40] = 220
        # Shadow to the right
        img[20:40, 40:60] = 5
        bbox = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}

        # Test with towfish altitude
        mens = calculate_shadow_mensuration(
            img, bbox,
            towfish_altitude_m=10.0,
            pixel_resolution_m=(0.5, 0.5),
            slant_range_m=30.0
        )
        self.assertGreater(mens["estimated_height_meters"], 0.1)
        self.assertGreater(mens["shadow_length_meters"], 0)
        self.assertTrue(mens["shadow_detected"])
        self.assertIn("target_length_meters", mens)
        self.assertIn("target_width_meters", mens)
        self.assertGreater(mens["target_length_meters"], 0.0)
        self.assertGreater(mens["target_width_meters"], 0.0)

    def test_slant_range_correction(self):
        """Verify slant-range correction geometry."""
        swath = np.full((50, 200), 50, dtype=np.uint8)
        # Simulate nadir water column
        swath[:, 85:115] = 5
        corrected, alt_px = correct_slant_range_waterfall(swath, altitude_px=15)
        self.assertEqual(corrected.shape[0], 50)
        self.assertGreater(corrected.shape[1], 50)

    def test_soft_nms(self):
        """Verify soft NMS decays overlapping candidate scores without immediate erasure."""
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52], # high overlap
            [70, 70, 90, 90]  # distinct
        ], dtype=float)
        scores = np.array([0.9, 0.85, 0.75], dtype=float)
        keep = soft_nms(boxes, scores, iou_threshold=0.45)
        self.assertIn(0, keep)
        self.assertIn(2, keep)

    def test_soft_nms_preserves_caller_boxes_immutability(self):
        """Verify soft_nms does not mutate the caller's boxes array in place and maps correctly."""
        boxes = np.array([
            [10, 10, 20, 20],
            [10, 10, 22, 22],
            [50, 50, 60, 60]
        ], dtype=float)
        scores = np.array([0.4, 0.9, 0.8], dtype=float)
        boxes_before = boxes.copy()

        keep = soft_nms(boxes, scores, iou_threshold=0.45)
        
        # Caller boxes array must not be mutated
        np.testing.assert_array_equal(boxes, boxes_before)
        # Highest score box (index 1, score 0.9) must be selected first
        self.assertEqual(keep[0], 1)
        np.testing.assert_array_equal(boxes[keep[0]], np.array([10, 10, 22, 22], dtype=float))

    def test_adaptive_backscatter_dilution(self):
        """Verify adaptive window isolates metallic highlights even when diluted by seabed (Fix 1)."""
        # 100x100 box where 75% is seabed (intensity 50) and 25% is metallic highlight (intensity 225)
        patch = np.full((100, 100), 50, dtype=np.uint8)
        patch[0:25, :] = 225  # 25% of the box
        bbox = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
        res = classify_material_density(patch, bbox)
        # Without adaptive window, P95 on 75% seabed might be diluted; with top 30% cluster isolation,
        # it extracts the bright cluster and flags as Hard (Metallic)
        self.assertEqual(res["material_density"], MATERIAL_HARD_METAL)
        self.assertGreaterEqual(res["peak_backscatter_p95"], 185.0)

    def test_bimodal_shadow_internal(self):
        """Verify bimodal shadow detection recognizes internal shadows (Kaggle style) (Fix 2)."""
        # Box has highlight on left (x: 0..20) and dark shadow on right (x: 20..50)
        img = np.full((60, 60), 100, dtype=np.uint8)
        img[10:50, 10:30] = 220  # Highlight
        img[10:50, 30:50] = 10   # Internal acoustic shadow (< 30 intensity)
        bbox = {"x1": 10, "y1": 10, "x2": 50, "y2": 50}

        mens = calculate_shadow_mensuration(img, bbox, pixel_resolution_m=(0.5, 0.5))
        self.assertTrue(mens["shadow_detected"])
        self.assertGreaterEqual(mens["shadow_length_pixels"], 10)
        self.assertGreater(mens["estimated_height_meters"], 0.1)

    def test_nominal_grazing_fallback_when_altitude_missing(self):
        """Verify nominal 15-deg grazing angle model fallback when towfish altitude is None (Fix 2)."""
        img = np.full((100, 100), 120, dtype=np.uint8)
        img[30:50, 30:50] = 230
        img[30:50, 50:75] = 15  # External shadow
        bbox = {"x1": 30, "y1": 30, "x2": 50, "y2": 50}

        mens = calculate_shadow_mensuration(img, bbox, towfish_altitude_m=None)
        self.assertEqual(mens["grazing_angle_deg"], 15.0)
        self.assertEqual(mens["mensuration_method"], "Nominal Hydrographic Grazing Model (15°)")
        self.assertGreater(mens["estimated_height_meters"], 0.0)

    def test_strict_tiled_mode_threshold(self):
        """Verify Fix 3: images > 1024px bypass the coarse global resized pass."""
        from unittest.mock import patch, MagicMock
        from backend.inference.engine import YOLOESIInferenceEngine

        engine = YOLOESIInferenceEngine()
        
        # Test 1: Image <= 1024 (e.g. 512x512) should invoke global pass with full img
        small_img = np.zeros((512, 512, 3), dtype=np.uint8)
        with patch.object(engine, "infer_single_image", return_value=(np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int))) as mock_infer:
            engine.infer_tiled(small_img, tile_size=256, overlap=64)
            calls = mock_infer.call_args_list
            last_arg = calls[-1][0][0]
            self.assertEqual(last_arg.shape, (512, 512, 3), "Images <= 1024 must include global pass")

        # Test 2: Image > 1024 (e.g. 1200x800) must NOT invoke global pass with full img
        large_img = np.zeros((1200, 800, 3), dtype=np.uint8)
        with patch.object(engine, "infer_single_image", return_value=(np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int))) as mock_infer:
            engine.infer_tiled(large_img, tile_size=256, overlap=64)
            calls_large = mock_infer.call_args_list
            for c in calls_large:
                arg_img = c[0][0]
                self.assertNotEqual(arg_img.shape, (1200, 800, 3), "Images > 1024 must bypass global pass")

if __name__ == "__main__":
    unittest.main()
