# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_tars.windows_runtime import (
    DPIAwareness,
    get_dpi_settings,
    MonitorManager,
    get_monitor_info,
    CoordinateMapper,
    CursorTracker,
    MouseCalibrator,
    DebugOverlay,
    ResolutionNormalizer,
)


class TestDPIAwareness(unittest.TestCase):
    def test_singleton_pattern(self):
        dpi1 = DPIAwareness()
        dpi2 = DPIAwareness()
        self.assertIs(dpi1, dpi2)

    def test_get_scale_factor(self):
        dpi = DPIAwareness()
        scale = dpi.get_scale_factor()
        self.assertIsInstance(scale, float)
        self.assertGreater(scale, 0)

    def test_logical_physical_conversion(self):
        dpi = DPIAwareness()
        logical_x, logical_y = 1000, 500
        physical_x, physical_y = dpi.logical_to_physical(logical_x, logical_y)
        self.assertIsInstance(physical_x, int)
        self.assertIsInstance(physical_y, int)

    def test_get_dpi_settings(self):
        settings = get_dpi_settings()
        self.assertIsNotNone(settings)
        self.assertIn(settings.awareness_name, [
            "DPI Unaware", "System DPI Aware", "Per-Monitor DPI Aware", "Per-Monitor v2 DPI Aware"
        ])


class TestMonitorManager(unittest.TestCase):
    def test_singleton_pattern(self):
        manager1 = MonitorManager()
        manager2 = MonitorManager()
        self.assertIs(manager1, manager2)

    def test_get_config(self):
        manager = MonitorManager()
        config = manager.get_config()
        self.assertIsNotNone(config)
        self.assertIsInstance(config.monitors, list)
        self.assertGreater(len(config.monitors), 0)

    def test_refresh(self):
        manager = MonitorManager()
        config = manager.refresh()
        self.assertIsNotNone(config)
        self.assertIsInstance(config.virtual_screen_width, int)

    def test_get_primary_monitor(self):
        manager = MonitorManager()
        primary = manager.get_primary_monitor()
        self.assertIsNotNone(primary)
        self.assertTrue(primary.is_primary)

    def test_get_monitor_info(self):
        info = get_monitor_info()
        self.assertIsNotNone(info)


class TestCoordinateMapper(unittest.TestCase):
    def test_initialization(self):
        mapper = CoordinateMapper()
        self.assertIsNotNone(mapper._dpi)
        self.assertIsNotNone(mapper._monitor_manager)

    def test_model_to_screen(self):
        mapper = CoordinateMapper()
        x_norm, y_norm = 0.5, 0.5
        screenshot_width, screenshot_height = 1920, 1080
        x, y = mapper.model_to_screen(x_norm, y_norm, screenshot_width, screenshot_height)
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)

    def test_normalize_coordinates(self):
        mapper = CoordinateMapper()
        config = mapper._monitor_manager.get_config()
        monitor = config.monitors[0]
        x_norm, y_norm = mapper.normalize_coordinates(monitor.x + 100, monitor.y + 100)
        self.assertGreaterEqual(x_norm, 0.0)
        self.assertLessEqual(x_norm, 1.0)
        self.assertGreaterEqual(y_norm, 0.0)
        self.assertLessEqual(y_norm, 1.0)

    def test_denormalize_to_screen(self):
        mapper = CoordinateMapper()
        x_norm, y_norm = 0.5, 0.5
        x, y = mapper.denormalize_to_screen(x_norm, y_norm)
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)

    def test_calibration_offset(self):
        mapper = CoordinateMapper()
        mapper.set_calibration_offset(10.5, -5.5)
        self.assertTrue(mapper._calibration_enabled)
        mapper.clear_calibration()
        self.assertFalse(mapper._calibration_enabled)


class TestCursorTracker(unittest.TestCase):
    def test_initialization(self):
        tracker = CursorTracker(drift_threshold=10.0)
        self.assertEqual(tracker._drift_threshold, 10.0)

    def test_get_current_position(self):
        tracker = CursorTracker()
        pos = tracker.get_current_position()
        self.assertIsNotNone(pos)
        self.assertIsInstance(pos.x, int)
        self.assertIsInstance(pos.y, int)
        self.assertIsInstance(pos.timestamp, float)

    def test_history_management(self):
        tracker = CursorTracker()
        tracker.clear_history()
        self.assertEqual(len(tracker.get_history()), 0)
        tracker.get_current_position()
        history = tracker.get_history()
        self.assertGreater(len(history), 0)

    def test_calibration_reset(self):
        tracker = CursorTracker()
        tracker._average_drift_x = 5.0
        tracker._average_drift_y = -3.0
        tracker._drift_samples = 10
        tracker.reset_calibration()
        self.assertEqual(tracker._average_drift_x, 0.0)
        self.assertEqual(tracker._average_drift_y, 0.0)
        self.assertEqual(tracker._drift_samples, 0)


class TestMouseCalibrator(unittest.TestCase):
    def test_initialization(self):
        calibrator = MouseCalibrator()
        self.assertIsNotNone(calibrator._cursor_tracker)
        self.assertEqual(len(calibrator._calibration_points), 5)

    def test_add_calibration_point(self):
        calibrator = MouseCalibrator()
        initial_count = len(calibrator._calibration_points)
        calibrator.add_calibration_point(100, 100)
        self.assertEqual(len(calibrator._calibration_points), initial_count + 1)

    def test_is_calibrated(self):
        calibrator = MouseCalibrator()
        self.assertFalse(calibrator.is_calibrated())
        calibrator._is_calibrated = True
        self.assertTrue(calibrator.is_calibrated())

    def test_quick_verify_no_calibration(self):
        calibrator = MouseCalibrator()
        is_accurate, error = calibrator.quick_verify()
        self.assertFalse(is_accurate)

    def test_apply_calibration_no_result(self):
        calibrator = MouseCalibrator()
        x, y = 500, 300
        calibrated_x, calibrated_y = calibrator.apply_calibration(x, y)
        self.assertEqual(calibrated_x, x)
        self.assertEqual(calibrated_y, y)


class TestDebugOverlay(unittest.TestCase):
    def test_initialization(self):
        overlay = DebugOverlay()
        self.assertFalse(overlay.is_enabled())

    def test_enable_disable(self):
        overlay = DebugOverlay()
        overlay.enable()
        self.assertTrue(overlay.is_enabled())
        overlay.disable()
        self.assertFalse(overlay.is_enabled())

    def test_record_transformation(self):
        overlay = DebugOverlay()
        info = overlay.record_transformation(
            x_norm=0.5,
            y_norm=0.5,
            screenshot_width=1920,
            screenshot_height=1080,
            cursor_x=960,
            cursor_y=540,
        )
        self.assertIsNotNone(info)
        self.assertIsInstance(info.model_predicted, tuple)
        self.assertIsInstance(info.translated, tuple)

    def test_history_management(self):
        overlay = DebugOverlay()
        overlay.clear_history()
        self.assertEqual(len(overlay.get_history()), 0)
        overlay.record_transformation(0.5, 0.5, 1920, 1080, 960, 540)
        self.assertGreater(len(overlay.get_history()), 0)

    def test_statistics(self):
        overlay = DebugOverlay()
        stats = overlay.get_statistics()
        self.assertIn("total_transforms", stats)
        self.assertIn("success_rate", stats)

    def test_test_coordinate_accuracy(self):
        overlay = DebugOverlay()
        test_points = [(100, 100), (500, 500), (1000, 500)]
        results = overlay.test_coordinate_accuracy(test_points, 1920, 1080)
        self.assertEqual(results["test_points"], 3)


class TestResolutionNormalizer(unittest.TestCase):
    def test_initialization(self):
        normalizer = ResolutionNormalizer()
        self.assertIsNotNone(normalizer._monitor_manager)

    def test_normalize_denormalize_roundtrip(self):
        normalizer = ResolutionNormalizer()
        config = normalizer._monitor_manager.get_config()
        monitor = config.monitors[0]
        original_x = monitor.x + monitor.width // 2
        original_y = monitor.y + monitor.height // 2
        normalized = normalizer.normalize(original_x, original_y)
        denormalized = normalizer.denormalize(normalized.x_norm, normalized.y_norm, normalized.monitor_index)
        tolerance = 2
        self.assertAlmostEqual(denormalized[0], original_x, delta=tolerance)
        self.assertAlmostEqual(denormalized[1], original_y, delta=tolerance)

    def test_normalize_to_screenshot_space(self):
        normalizer = ResolutionNormalizer()
        config = normalizer._monitor_manager.get_config()
        monitor = config.monitors[0]
        center_x = monitor.x + monitor.width // 2
        center_y = monitor.y + monitor.height // 2
        x_model, y_model = normalizer.normalize_to_screenshot_space(
            center_x, center_y, monitor.width, monitor.height
        )
        self.assertGreaterEqual(x_model, 0.0)
        self.assertLessEqual(x_model, 1.0)
        self.assertGreaterEqual(y_model, 0.0)
        self.assertLessEqual(y_model, 1.0)

    def test_aspect_ratio_offsets(self):
        normalizer = ResolutionNormalizer()
        offsets = normalizer.calculate_aspect_ratio_offsets(1920, 1080)
        self.assertEqual(len(offsets), 4)
        self.assertIsInstance(offsets[0], int)
        self.assertIsInstance(offsets[1], int)

    def test_get_current_profile(self):
        normalizer = ResolutionNormalizer()
        profile = normalizer.get_current_profile()
        if profile:
            self.assertIsNotNone(profile.width)
            self.assertIsNotNone(profile.height)
            self.assertGreater(profile.width, 0)
            self.assertGreater(profile.height, 0)


if __name__ == "__main__":
    unittest.main()