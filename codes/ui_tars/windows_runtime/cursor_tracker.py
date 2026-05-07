# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes


@dataclass
class CursorPosition:
    x: int
    y: int
    timestamp: float
    is_valid: bool = True


@dataclass
class CursorDriftInfo:
    expected_x: float
    expected_y: float
    actual_x: int
    actual_y: int
    drift_x: float
    drift_y: float
    drift_magnitude: float
    is_within_threshold: bool


class CursorTracker:
    def __init__(self, drift_threshold: float = 5.0):
        self._drift_threshold = drift_threshold
        self._history: list[CursorPosition] = []
        self._max_history = 100
        self._last_click_position: Optional[CursorPosition] = None
        self._average_drift_x: float = 0.0
        self._average_drift_y: float = 0.0
        self._drift_samples: int = 0

    def get_current_position(self) -> CursorPosition:
        if sys.platform != "win32":
            return CursorPosition(0, 0, time.time(), False)

        try:
            user32 = ctypes.windll.user32
            cursor = wintypes.POINT()
            if user32.GetCursorPos(ctypes.byref(cursor)):
                pos = CursorPosition(
                    x=cursor.x, y=cursor.y, timestamp=time.time(), is_valid=True
                )
                self._add_to_history(pos)
                return pos
        except Exception:
            pass
        return CursorPosition(0, 0, time.time(), False)

    def _add_to_history(self, position: CursorPosition) -> None:
        self._history.append(position)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def set_click_target(self, x: int, y: int) -> None:
        self._last_click_position = CursorPosition(x, y, time.time(), True)

    def verify_click(self, timeout: float = 0.5) -> CursorDriftInfo:
        time.sleep(timeout)
        actual = self.get_current_position()

        if self._last_click_position is None:
            return CursorDriftInfo(
                expected_x=0,
                expected_y=0,
                actual_x=actual.x,
                actual_y=actual.y,
                drift_x=0,
                drift_y=0,
                drift_magnitude=0,
                is_within_threshold=True,
            )

        expected = self._last_click_position
        drift_x = actual.x - expected.x
        drift_y = actual.y - expected.y
        drift_magnitude = (drift_x**2 + drift_y**2) ** 0.5

        self._update_drift_averages(drift_x, drift_y)

        return CursorDriftInfo(
            expected_x=expected.x,
            expected_y=expected.y,
            actual_x=actual.x,
            actual_y=actual.y,
            drift_x=drift_x,
            drift_y=drift_y,
            drift_magnitude=drift_magnitude,
            is_within_threshold=drift_magnitude <= self._drift_threshold,
        )

    def _update_drift_averages(self, drift_x: float, drift_y: float) -> None:
        self._drift_samples += 1
        alpha = 0.3
        if self._drift_samples == 1:
            self._average_drift_x = drift_x
            self._average_drift_y = drift_y
        else:
            self._average_drift_x = alpha * drift_x + (1 - alpha) * self._average_drift_x
            self._average_drift_y = alpha * drift_y + (1 - alpha) * self._average_drift_y

    def get_calibration_offset(self) -> tuple[float, float]:
        return (self._average_drift_x, self._average_drift_y)

    def reset_calibration(self) -> None:
        self._average_drift_x = 0.0
        self._average_drift_y = 0.0
        self._drift_samples = 0

    def get_history(self) -> list[CursorPosition]:
        return self._history.copy()

    def clear_history(self) -> None:
        self._history.clear()

    def wait_for_cursor_stable(
        self, x: int, y: int, threshold: float = 2.0, max_wait: float = 2.0
    ) -> bool:
        start_time = time.time()
        stable_count = 0
        required_stable = 3

        while time.time() - start_time < max_wait:
            current = self.get_current_position()
            if abs(current.x - x) <= threshold and abs(current.y - y) <= threshold:
                stable_count += 1
                if stable_count >= required_stable:
                    return True
            else:
                stable_count = 0
            time.sleep(0.05)
        return False

    def track_cursor_during_move(
        self,
        target_x: int,
        target_y: int,
        callback: Optional[Callable[[int, int, float], None]] = None,
        samples: int = 10,
    ) -> list[tuple[int, int, float]]:
        positions = []
        for i in range(samples):
            pos = self.get_current_position()
            progress = i / samples
            positions.append((pos.x, pos.y, progress))
            if callback:
                callback(pos.x, pos.y, progress)
            time.sleep(0.05)
        return positions