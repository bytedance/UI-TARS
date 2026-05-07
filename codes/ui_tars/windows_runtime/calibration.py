# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from dataclasses import dataclass
from typing import Optional

if sys.platform == "win32":
    import pyautogui

from .cursor_tracker import CursorTracker


@dataclass
class CalibrationPoint:
    target_x: int
    target_y: int
    actual_x: int
    actual_y: int
    error_x: float
    error_y: float
    error_magnitude: float
    success: bool


@dataclass
class CalibrationResult:
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float
    rotation: float
    confidence: float
    sample_count: int
    points: list[CalibrationPoint]


class MouseCalibrator:
    def __init__(self, cursor_tracker: Optional[CursorTracker] = None):
        self._cursor_tracker = cursor_tracker or CursorTracker()
        self._calibration_points: list[tuple[int, int]] = [
            (960, 540),
            (192, 108),
            (1728, 972),
            (192, 972),
            (1728, 108),
        ]
        self._calibration_result: Optional[CalibrationResult] = None
        self._is_calibrated: bool = False

    def set_calibration_points(self, points: list[tuple[int, int]]) -> None:
        if len(points) >= 3:
            self._calibration_points = points

    def add_calibration_point(self, x: int, y: int) -> None:
        self._calibration_points.append((x, y))

    def run_calibration(self) -> CalibrationResult:
        if sys.platform != "win32":
            return self._create_default_result()

        results = []
        for target_x, target_y in self._calibration_points:
            self._cursor_tracker.set_click_target(target_x, target_y)
            pyautogui.moveTo(target_x, target_y)
            time.sleep(0.1)
            actual = self._cursor_tracker.get_current_position()

            error_x = actual.x - target_x
            error_y = actual.y - target_y
            error_magnitude = (error_x**2 + error_y**2) ** 0.5

            results.append(
                CalibrationPoint(
                    target_x=target_x,
                    target_y=target_y,
                    actual_x=actual.x,
                    actual_y=actual.y,
                    error_x=error_x,
                    error_y=error_y,
                    error_magnitude=error_magnitude,
                    success=error_magnitude < 10.0,
                )
            )

        success_count = sum(1 for p in results if p.success)
        confidence = success_count / len(results) if results else 0.0

        avg_error_x = sum(p.error_x for p in results) / len(results) if results else 0.0
        avg_error_y = sum(p.error_y for p in results) / len(results) if results else 0.0

        self._calibration_result = CalibrationResult(
            offset_x=-avg_error_x,
            offset_y=-avg_error_y,
            scale_x=1.0,
            scale_y=1.0,
            rotation=0.0,
            confidence=confidence,
            sample_count=len(results),
            points=results,
        )
        self._is_calibrated = True
        return self._calibration_result

    def _create_default_result(self) -> CalibrationResult:
        return CalibrationResult(
            offset_x=0.0,
            offset_y=0.0,
            scale_x=1.0,
            scale_y=1.0,
            rotation=0.0,
            confidence=0.0,
            sample_count=0,
            points=[],
        )

    def apply_calibration(self, x: int, y: int) -> tuple[int, int]:
        if not self._is_calibrated or self._calibration_result is None:
            return (x, y)
        result = self._calibration_result
        calibrated_x = int((x + result.offset_x) * result.scale_x)
        calibrated_y = int((y + result.offset_y) * result.scale_y)
        return (calibrated_x, calibrated_y)

    def get_calibration_result(self) -> Optional[CalibrationResult]:
        return self._calibration_result

    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def reset_calibration(self) -> None:
        self._calibration_result = None
        self._is_calibrated = False

    def quick_verify(self) -> tuple[bool, float]:
        if not self._is_calibrated:
            return (False, 0.0)

        x, y = 960, 540
        calibrated = self.apply_calibration(x, y)
        error = ((calibrated[0] - x) ** 2 + (calibrated[1] - y) ** 2) ** 0.5
        return (error < 5.0, error)

    def auto_calibrate(
        self, max_attempts: int = 3, target_accuracy: float = 2.0
    ) -> CalibrationResult:
        for attempt in range(max_attempts):
            result = self.run_calibration()
            if result.confidence >= 0.8:
                return result
            if attempt < max_attempts - 1:
                time.sleep(0.5)
        return result