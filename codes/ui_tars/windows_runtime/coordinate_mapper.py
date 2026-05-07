# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from typing import Optional

from .dpi import DPIAwareness
from .monitor_manager import MonitorManager, MonitorInfo, DisplayConfig


@dataclass
class CoordinateTransform:
    source_space: str
    target_space: str
    x: int
    y: int
    monitor_name: Optional[str] = None
    scale_factor: float = 1.0
    offset_x: int = 0
    offset_y: int = 0


class CoordinateMapper:
    def __init__(
        self,
        dpi_awareness: Optional[DPIAwareness] = None,
        monitor_manager: Optional[MonitorManager] = None,
    ):
        self._dpi = dpi_awareness or DPIAwareness()
        self._monitor_manager = monitor_manager or MonitorManager()
        self._calibration_offset_x: float = 0.0
        self._calibration_offset_y: float = 0.0
        self._calibration_enabled: bool = False

    def set_calibration_offset(self, offset_x: float, offset_y: float) -> None:
        self._calibration_offset_x = offset_x
        self._calibration_offset_y = offset_y
        self._calibration_enabled = True

    def clear_calibration(self) -> None:
        self._calibration_offset_x = 0.0
        self._calibration_offset_y = 0.0
        self._calibration_enabled = False

    def model_to_screen(
        self,
        x_norm: float,
        y_norm: float,
        screenshot_width: int,
        screenshot_height: int,
        monitor_index: int = 0,
    ) -> tuple[int, int]:
        config = self._monitor_manager.get_config()
        if monitor_index >= len(config.monitors):
            monitor_index = config.primary_monitor_index
        monitor = config.monitors[monitor_index]

        screen_x = int(x_norm * screenshot_width)
        screen_y = int(y_norm * screenshot_height)

        aspect_ratio_screen = screenshot_width / screenshot_height
        aspect_ratio_monitor = monitor.width / monitor.height

        if aspect_ratio_screen > aspect_ratio_monitor:
            new_width = monitor.width
            new_height = int(monitor.width / aspect_ratio_screen)
            offset_x = 0
            offset_y = (monitor.height - new_height) // 2
        else:
            new_height = monitor.height
            new_width = int(monitor.height * aspect_ratio_screen)
            offset_x = (monitor.width - new_width) // 2
            offset_y = 0

        final_x = monitor.x + offset_x + screen_x
        final_y = monitor.y + offset_y + screen_y

        if self._calibration_enabled:
            final_x = int(final_x + self._calibration_offset_x)
            final_y = int(final_y + self._calibration_offset_y)

        return (final_x, final_y)

    def model_to_screen_with_dpi(
        self,
        x_norm: float,
        y_norm: float,
        screenshot_width: int,
        screenshot_height: int,
        monitor_index: int = 0,
    ) -> tuple[int, int]:
        x, y = self.model_to_screen(
            x_norm, y_norm, screenshot_width, screenshot_height, monitor_index
        )

        if sys.platform == "win32":
            dpi = self._dpi
            logical_x, logical_y = dpi.physical_to_logical(x, y)
            return (logical_x, logical_y)
        return (x, y)

    def screen_to_monitor(
        self, x: int, y: int
    ) -> tuple[Optional[MonitorInfo], tuple[int, int]]:
        monitor = self._monitor_manager.get_monitor_at_point(x, y)
        if monitor:
            local_x = x - monitor.x
            local_y = y - monitor.y
            return (monitor, (local_x, local_y))
        return (None, (x, y))

    def normalize_coordinates(
        self, x: int, y: int, monitor_index: Optional[int] = None
    ) -> tuple[float, float]:
        config = self._monitor_manager.get_config()
        if monitor_index is not None and 0 <= monitor_index < len(config.monitors):
            monitor = config.monitors[monitor_index]
        else:
            monitor_result = self._monitor_manager.get_monitor_at_point(x, y)
            if monitor_result is None:
                return (0.0, 0.0)
            monitor = monitor_result

        x_norm = (x - monitor.x) / monitor.width
        y_norm = (y - monitor.y) / monitor.height
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        return (x_norm, y_norm)

    def denormalize_to_screen(
        self,
        x_norm: float,
        y_norm: float,
        monitor_index: int = 0,
    ) -> tuple[int, int]:
        config = self._monitor_manager.get_config()
        if monitor_index >= len(config.monitors):
            monitor_index = config.primary_monitor_index

        monitor = config.monitors[monitor_index]
        x = int(monitor.x + x_norm * monitor.width)
        y = int(monitor.y + y_norm * monitor.height)
        return (x, y)

    def denormalize_to_physical(
        self,
        x_norm: float,
        y_norm: float,
        monitor_index: int = 0,
    ) -> tuple[int, int]:
        x, y = self.denormalize_to_screen(x_norm, y_norm, monitor_index)

        if sys.platform == "win32":
            dpi = self._dpi
            return dpi.logical_to_physical(x, y)
        return (x, y)

    def get_transform_info(
        self,
        x_norm: float,
        y_norm: float,
        screenshot_width: int,
        screenshot_height: int,
        monitor_index: int = 0,
    ) -> CoordinateTransform:
        config = self._monitor_manager.get_config()
        if monitor_index >= len(config.monitors):
            monitor_index = config.primary_monitor_index
        monitor = config.monitors[monitor_index]

        screen_coords = self.model_to_screen(
            x_norm, y_norm, screenshot_width, screenshot_height, monitor_index
        )

        return CoordinateTransform(
            source_space="normalized (0-1)",
            target_space=f"screen ({monitor.name})",
            x=screen_coords[0],
            y=screen_coords[1],
            monitor_name=monitor.name,
            scale_factor=monitor.scale_factor,
            offset_x=monitor.x,
            offset_y=monitor.y,
        )

    def verify_click_target(
        self, x: int, y: int, expected_x: int, expected_y: int
    ) -> tuple[bool, float]:
        error = ((x - expected_x) ** 2 + (y - expected_y) ** 2) ** 0.5
        threshold = 5.0
        return (error <= threshold, error)