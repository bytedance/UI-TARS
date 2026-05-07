# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from typing import Optional

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes


@dataclass
class DPISettings:
    awareness_level: int
    awareness_name: str
    dpi: int
    scale_factor: float
    is_per_monitor_aware: bool


class DPIAwareness:
    AWARENESS_UNAWARE = 0
    AWARENESS_SYSTEM = 1
    AWARENESS_PER_MONITOR = 2
    AWARENESS_PER_MONITOR_V2 = 3

    _instance: Optional["DPIAwareness"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if DPIAwareness._initialized:
            return
        DPIAwareness._initialized = True
        self._dpi: int = 96
        self._scale_factor: float = 1.0
        self._awareness_level: int = self.AWARENESS_UNAWARE
        self._is_per_monitor_aware: bool = False
        if sys.platform == "win32":
            self._setup_dpi_awareness()

    def _setup_dpi_awareness(self) -> None:
        try:
            shcore = ctypes.windll.shcore
            user32 = ctypes.windll.user32

            PROCESS_PER_MONITOR_DPI_AWARE = 2
            result = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
            if result == 0:
                self._awareness_level = self.AWARENESS_PER_MONITOR
                self._is_per_monitor_aware = True
            else:
                try:
                    user32.SetProcessDPIAware()
                    self._awareness_level = self.AWARENESS_SYSTEM
                    self._is_per_monitor_aware = False
                except Exception:
                    self._awareness_level = self.AWARENESS_UNAWARE
                    self._is_per_monitor_aware = False

            self._update_dpi()
        except Exception:
            self._awareness_level = self.AWARENESS_UNAWARE
            self._is_per_monitor_aware = False

    def _update_dpi(self) -> None:
        if sys.platform != "win32":
            return
        try:
            user32 = ctypes.windll.user32
            self._dpi = user32.GetDpiForSystem()
            self._scale_factor = self._dpi / 96.0
        except Exception:
            self._dpi = 96
            self._scale_factor = 1.0

    def get_dpi(self) -> int:
        if sys.platform == "win32":
            self._update_dpi()
        return self._dpi

    def get_scale_factor(self) -> float:
        if sys.platform == "win32":
            self._update_dpi()
        return self._scale_factor

    def get_awareness_level(self) -> int:
        return self._awareness_level

    def get_awareness_name(self) -> str:
        names = {
            self.AWARENESS_UNAWARE: "DPI Unaware",
            self.AWARENESS_SYSTEM: "System DPI Aware",
            self.AWARENESS_PER_MONITOR: "Per-Monitor DPI Aware",
            self.AWARENESS_PER_MONITOR_V2: "Per-Monitor v2 DPI Aware",
        }
        return names.get(self._awareness_level, "Unknown")

    def logical_to_physical(self, x: int, y: int) -> tuple[int, int]:
        scale = self.get_scale_factor()
        return (int(x * scale), int(y * scale))

    def physical_to_logical(self, x: int, y: int) -> tuple[int, int]:
        scale = self.get_scale_factor()
        if scale == 0:
            scale = 1.0
        return (int(x / scale), int(y / scale))


def get_dpi_settings() -> DPISettings:
    dpi_awareness = DPIAwareness()
    awareness_level = dpi_awareness.get_awareness_level()
    awareness_names = {
        0: "DPI Unaware",
        1: "System DPI Aware",
        2: "Per-Monitor DPI Aware",
        3: "Per-Monitor v2 DPI Aware",
    }
    return DPISettings(
        awareness_level=awareness_level,
        awareness_name=awareness_names.get(awareness_level, "Unknown"),
        dpi=dpi_awareness.get_dpi(),
        scale_factor=dpi_awareness.get_scale_factor(),
        is_per_monitor_aware=dpi_awareness._is_per_monitor_aware,
    )