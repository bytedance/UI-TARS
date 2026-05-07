# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass, field
from typing import Optional

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes


@dataclass
class MonitorInfo:
    handle: int
    name: str
    x: int
    y: int
    width: int
    height: int
    work_x: int
    work_y: int
    work_width: int
    work_height: int
    dpi: int
    scale_factor: float
    is_primary: bool
    is_virtual: bool = False


@dataclass
class DisplayConfig:
    monitors: list[MonitorInfo] = field(default_factory=list)
    virtual_screen_x: int = 0
    virtual_screen_y: int = 0
    virtual_screen_width: int = 0
    virtual_screen_height: int = 0
    primary_monitor_index: int = 0


class MonitorManager:
    _instance: Optional["MonitorManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized") or not self._initialized:
            self._initialized = True
            self._config: Optional[DisplayConfig] = None
            self._cache_valid: bool = False

    def _get_monitor_enum_callback(self, hMonitor: int, hdcMonitor: int, lParam: int) -> int:
        return 1

    def refresh(self) -> DisplayConfig:
        if sys.platform != "win32":
            return self._get_default_config()

        try:
            monitors = []
            primary_found = False
            primary_index = 0

            MONITORENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )

            user32 = ctypes.windll.user32

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", wintypes.LONG),
                    ("top", wintypes.LONG),
                    ("right", wintypes.LONG),
                    ("bottom", wintypes.LONG),
                ]

            class MONITORINFOEX(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", RECT),
                    ("rcWork", RECT),
                    ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32),
                ]

            def enum_callback(hMonitor, hdc, lParam):
                info = MONITORINFOEX()
                info.cbSize = ctypes.sizeof(MONITORINFOEX)
                if user32.GetMonitorInfoW(hMonitor, ctypes.byref(info)):
                    name = info.szDevice
                    is_primary = bool(info.dwFlags & 1)

                    dpi = 96
                    try:
                        shcore = ctypes.windll.shcore
                        dpiX = ctypes.c_uint()
                        dpiY = ctypes.c_uint()
                        shcore.GetDpiForMonitor(
                            hMonitor, 0, ctypes.byref(dpiX), ctypes.byref(dpiY)
                        )
                        dpi = dpiX.value
                    except Exception:
                        dpi = user32.GetDpiForSystem()

                    scale_factor = dpi / 96.0

                    monitor = MonitorInfo(
                        handle=hMonitor,
                        name=name,
                        x=info.rcMonitor.left,
                        y=info.rcMonitor.top,
                        width=info.rcMonitor.right - info.rcMonitor.left,
                        height=info.rcMonitor.bottom - info.rcMonitor.top,
                        work_x=info.rcWork.left,
                        work_y=info.rcWork.top,
                        work_width=info.rcWork.right - info.rcWork.left,
                        work_height=info.rcWork.bottom - info.rcWork.top,
                        dpi=dpi,
                        scale_factor=scale_factor,
                        is_primary=is_primary,
                    )
                    monitors.append(monitor)

                    if is_primary:
                        primary_index = len(monitors) - 1
                return 1

            user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(enum_callback), 0)

            if not monitors:
                return self._get_default_config()

            virtual_screen_x = min(m.x for m in monitors)
            virtual_screen_y = min(m.y for m in monitors)
            virtual_screen_width = max(m.x + m.width for m in monitors) - virtual_screen_x
            virtual_screen_height = max(m.y + m.height for m in monitors) - virtual_screen_y

            self._config = DisplayConfig(
                monitors=monitors,
                virtual_screen_x=virtual_screen_x,
                virtual_screen_y=virtual_screen_y,
                virtual_screen_width=virtual_screen_width,
                virtual_screen_height=virtual_screen_height,
                primary_monitor_index=primary_index,
            )
            self._cache_valid = True
            return self._config

        except Exception:
            return self._get_default_config()

    def _get_default_config(self) -> DisplayConfig:
        return DisplayConfig(
            monitors=[
                MonitorInfo(
                    handle=0,
                    name="DISPLAY",
                    x=0,
                    y=0,
                    width=1920,
                    height=1080,
                    work_x=0,
                    work_y=0,
                    work_width=1920,
                    work_height=1080,
                    dpi=96,
                    scale_factor=1.0,
                    is_primary=True,
                )
            ],
            virtual_screen_x=0,
            virtual_screen_y=0,
            virtual_screen_width=1920,
            virtual_screen_height=1080,
            primary_monitor_index=0,
        )

    def get_config(self, force_refresh: bool = False) -> DisplayConfig:
        if force_refresh or not self._cache_valid or self._config is None:
            return self.refresh()
        return self._config

    def get_monitor_at_point(self, x: int, y: int) -> Optional[MonitorInfo]:
        config = self.get_config()
        for monitor in config.monitors:
            if (
                monitor.x <= x < monitor.x + monitor.width
                and monitor.y <= y < monitor.y + monitor.height
            ):
                return monitor
        return config.monitors[config.primary_monitor_index] if config.monitors else None

    def get_monitor_at_point_normalized(
        self, x_norm: float, y_norm: float, monitor_index: int = 0
    ) -> tuple[int, int]:
        config = self.get_config()
        if monitor_index >= len(config.monitors):
            monitor_index = config.primary_monitor_index
        monitor = config.monitors[monitor_index]
        x = int(x_norm * monitor.width) + monitor.x
        y = int(y_norm * monitor.height) + monitor.y
        return (x, y)

    def get_primary_monitor(self) -> Optional[MonitorInfo]:
        config = self.get_config()
        if config.monitors:
            return config.monitors[config.primary_monitor_index]
        return None


def get_monitor_info() -> DisplayConfig:
    manager = MonitorManager()
    return manager.get_config(force_refresh=True)