# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from .monitor_manager import MonitorManager, DisplayConfig


@dataclass
class NormalizedPoint:
    x_norm: float
    y_norm: float
    monitor_index: int


@dataclass
class ResolutionProfile:
    name: str
    width: int
    height: int
    dpi: int
    scale_factor: float


class ResolutionNormalizer:
    def __init__(self, monitor_manager: Optional[MonitorManager] = None):
        self._monitor_manager = monitor_manager or MonitorManager()

    def normalize(
        self, x: int, y: int, monitor_index: Optional[int] = None
    ) -> NormalizedPoint:
        config = self._monitor_manager.get_config()

        if monitor_index is not None and 0 <= monitor_index < len(config.monitors):
            monitor = config.monitors[monitor_index]
        else:
            monitor = self._monitor_manager.get_monitor_at_point(x, y)
            if monitor is None:
                return NormalizedPoint(
                    x_norm=0.0, y_norm=0.0, monitor_index=0
                )

        x_norm = (x - monitor.x) / monitor.width
        y_norm = (y - monitor.y) / monitor.height
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        monitor_idx = config.monitors.index(monitor) if monitor in config.monitors else 0

        return NormalizedPoint(
            x_norm=x_norm,
            y_norm=y_norm,
            monitor_index=monitor_idx,
        )

    def denormalize(
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

    def denormalize_to_virtual_screen(
        self,
        x_norm: float,
        y_norm: float,
        monitor_index: int = 0,
    ) -> tuple[int, int]:
        config = self._monitor_manager.get_config()
        virtual_x = config.virtual_screen_x + int(x_norm * config.virtual_screen_width)
        virtual_y = config.virtual_screen_y + int(y_norm * config.virtual_screen_height)
        return (virtual_x, virtual_y)

    def normalize_to_screenshot_space(
        self,
        x: int,
        y: int,
        screenshot_width: int,
        screenshot_height: int,
        monitor_index: int = 0,
    ) -> tuple[float, float]:
        config = self._monitor_manager.get_config()

        if monitor_index >= len(config.monitors):
            monitor_index = config.primary_monitor_index

        monitor = config.monitors[monitor_index]

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

        local_x = x - monitor.x - offset_x
        local_y = y - monitor.y - offset_y

        x_model = local_x / new_width if new_width > 0 else 0.0
        y_model = local_y / new_height if new_height > 0 else 0.0

        x_model = max(0.0, min(1.0, x_model))
        y_model = max(0.0, min(1.0, y_model))

        return (x_model, y_model)

    def get_current_profile(self, monitor_index: int = 0) -> Optional[ResolutionProfile]:
        config = self._monitor_manager.get_config()

        if monitor_index >= len(config.monitors):
            return None

        monitor = config.monitors[monitor_index]

        return ResolutionProfile(
            name=monitor.name,
            width=monitor.width,
            height=monitor.height,
            dpi=monitor.dpi,
            scale_factor=monitor.scale_factor,
        )

    def calculate_aspect_ratio_offsets(
        self,
        screenshot_width: int,
        screenshot_height: int,
        monitor_index: int = 0,
    ) -> tuple[int, int, int, int]:
        config = self._monitor_manager.get_config()

        if monitor_index >= len(config.monitors):
            monitor_index = config.primary_monitor_index

        monitor = config.monitors[monitor_index]

        aspect_ratio_screen = screenshot_width / screenshot_height
        aspect_ratio_monitor = monitor.width / monitor.height

        if aspect_ratio_screen > aspect_ratio_monitor:
            render_width = monitor.width
            render_height = int(monitor.width / aspect_ratio_screen)
            offset_x = 0
            offset_y = (monitor.height - render_height) // 2
        else:
            render_height = monitor.height
            render_width = int(monitor.height * aspect_ratio_screen)
            offset_x = (monitor.width - render_width) // 2
            offset_y = 0

        return (render_width, render_height, offset_x, offset_y)

    def is_resolution_supported(
        self, width: int, height: int, monitor_index: int = 0
    ) -> bool:
        profile = self.get_current_profile(monitor_index)
        if profile is None:
            return False
        return width <= profile.width and height <= profile.height