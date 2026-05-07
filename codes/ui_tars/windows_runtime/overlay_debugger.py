# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from typing import Optional, Callable

from .monitor_manager import MonitorManager, DisplayConfig
from .coordinate_mapper import CoordinateMapper


@dataclass
class DebugInfo:
    model_predicted: tuple[float, float]
    translated: tuple[int, int]
    actual_cursor: tuple[int, int]
    error: float
    monitor_name: str
    dpi_scale: float
    screen_size: tuple[int, int]
    is_within_threshold: bool


class DebugOverlay:
    def __init__(
        self,
        monitor_manager: Optional[MonitorManager] = None,
        coordinate_mapper: Optional[CoordinateMapper] = None,
    ):
        self._monitor_manager = monitor_manager or MonitorManager()
        self._coordinate_mapper = coordinate_mapper or CoordinateMapper()
        self._enabled: bool = False
        self._log_callback: Optional[Callable[[str], None]] = None
        self._debug_history: list[DebugInfo] = []
        self._max_history = 50

    def set_log_callback(self, callback: Callable[[str], None]) -> None:
        self._log_callback = callback

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(message)
        else:
            print(f"[DebugOverlay] {message}")

    def enable(self) -> None:
        self._enabled = True
        self._log("Debug overlay enabled")

    def disable(self) -> None:
        self._enabled = False
        self._log("Debug overlay disabled")

    def is_enabled(self) -> bool:
        return self._enabled

    def record_transformation(
        self,
        x_norm: float,
        y_norm: float,
        screenshot_width: int,
        screenshot_height: int,
        cursor_x: int,
        cursor_y: int,
        monitor_index: int = 0,
    ) -> DebugInfo:
        config = self._monitor_manager.get_config()
        if monitor_index >= len(config.monitors):
            monitor_index = config.primary_monitor_index
        monitor = config.monitors[monitor_index]

        translated = self._coordinate_mapper.model_to_screen(
            x_norm, y_norm, screenshot_width, screenshot_height, monitor_index
        )

        error = ((translated[0] - cursor_x) ** 2 + (translated[1] - cursor_y) ** 2) ** 0.5
        threshold = 5.0

        debug_info = DebugInfo(
            model_predicted=(x_norm, y_norm),
            translated=translated,
            actual_cursor=(cursor_x, cursor_y),
            error=error,
            monitor_name=monitor.name,
            dpi_scale=monitor.scale_factor,
            screen_size=(monitor.width, monitor.height),
            is_within_threshold=error <= threshold,
        )

        self._add_to_history(debug_info)

        if self._enabled:
            self._log_transform(debug_info)

        return debug_info

    def _add_to_history(self, info: DebugInfo) -> None:
        self._debug_history.append(info)
        if len(self._debug_history) > self._max_history:
            self._debug_history.pop(0)

    def _log_transform(self, info: DebugInfo) -> None:
        status = "OK" if info.is_within_threshold else "ERROR"
        self._log(f"[{status}] MODEL: ({info.model_predicted[0]:.4f}, {info.model_predicted[1]:.4f})")
        self._log(f"       TRANSLATED: {info.translated}")
        self._log(f"       ACTUAL:     {info.actual_cursor}")
        self._log(f"       ERROR:      {info.error:.2f}px")
        self._log(f"       MONITOR:    {info.monitor_name} @ {info.dpi_scale:.2f}x")

    def get_last_debug_info(self) -> Optional[DebugInfo]:
        return self._debug_history[-1] if self._debug_history else None

    def get_history(self) -> list[DebugInfo]:
        return self._debug_history.copy()

    def clear_history(self) -> None:
        self._debug_history.clear()

    def get_statistics(self) -> dict:
        if not self._debug_history:
            return {
                "total_transforms": 0,
                "successful_transforms": 0,
                "failed_transforms": 0,
                "success_rate": 0.0,
                "average_error": 0.0,
                "max_error": 0.0,
            }

        successful = sum(1 for info in self._debug_history if info.is_within_threshold)
        total = len(self._debug_history)
        errors = [info.error for info in self._debug_history]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)

        return {
            "total_transforms": total,
            "successful_transforms": successful,
            "failed_transforms": total - successful,
            "success_rate": successful / total,
            "average_error": avg_error,
            "max_error": max_error,
        }

    def create_test_report(self) -> str:
        stats = self.get_statistics()
        lines = [
            "=" * 60,
            "Windows Runtime Debug Report",
            "=" * 60,
            f"Total Transforms: {stats['total_transforms']}",
            f"Successful: {stats['successful_transforms']}",
            f"Failed: {stats['failed_transforms']}",
            f"Success Rate: {stats['success_rate'] * 100:.1f}%",
            f"Average Error: {stats['average_error']:.2f}px",
            f"Max Error: {stats['max_error']:.2f}px",
            "=" * 60,
        ]

        if self._debug_history:
            lines.append("\nRecent Transforms:")
            for i, info in enumerate(self._debug_history[-5:]):
                status = "OK" if info.is_within_threshold else "FAIL"
                lines.append(
                    f"  [{status}] ({info.model_predicted[0]:.3f}, {info.model_predicted[1]:.3f}) "
                    f"-> {info.translated} (err: {info.error:.1f}px)"
                )

        return "\n".join(lines)

    def test_coordinate_accuracy(
        self,
        test_points: list[tuple[int, int]],
        screenshot_width: int,
        screenshot_height: int,
    ) -> dict:
        results = []
        for target_x, target_y in test_points:
            x_norm = target_x / screenshot_width
            y_norm = target_y / screenshot_height

            translated = self._coordinate_mapper.model_to_screen(
                x_norm, y_norm, screenshot_width, screenshot_height
            )

            error = ((translated[0] - target_x) ** 2 + (translated[1] - target_y) ** 2) ** 0.5
            results.append({
                "target": (target_x, target_y),
                "translated": translated,
                "error": error,
                "accurate": error < 5.0,
            })

        accurate_count = sum(1 for r in results if r["accurate"])
        return {
            "test_points": len(test_points),
            "accurate_points": accurate_count,
            "accuracy": accurate_count / len(test_points) if results else 0.0,
            "results": results,
        }