# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Windows Runtime Integration Example

This module demonstrates how to integrate the Windows Compatibility Layer
with UI-TARS for improved coordinate accuracy on Windows systems.
"""

import sys

if sys.platform != "win32":
    print("This module is designed for Windows systems only.")
    sys.exit(0)

from ui_tars.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code
from ui_tars.windows_runtime import (
    CoordinateMapper,
    CursorTracker,
    MouseCalibrator,
    DebugOverlay,
    MonitorManager,
    DPIAwareness,
)


class WindowsAwareAgent:
    def __init__(self, enable_calibration: bool = False, debug: bool = False):
        self._dpi = DPIAwareness()
        self._monitor_manager = MonitorManager()
        self._coordinate_mapper = CoordinateMapper(
            dpi_awareness=self._dpi,
            monitor_manager=self._monitor_manager,
        )
        self._cursor_tracker = CursorTracker()
        self._debug_overlay = DebugOverlay(
            monitor_manager=self._monitor_manager,
            coordinate_mapper=self._coordinate_mapper,
        )

        self._calibrator = None
        if enable_calibration:
            self._calibrator = MouseCalibrator(cursor_tracker=self._cursor_tracker)
            self._calibrator.run_calibration()
            offset_x, offset_y = self._calibrator._calibration_result.offset_x, self._calibrator._calibration_result.offset_y
            self._coordinate_mapper.set_calibration_offset(offset_x, offset_y)

        if debug:
            self._debug_overlay.enable()

    def process_model_response(
        self,
        model_response: str,
        screenshot_width: int,
        screenshot_height: int,
    ) -> tuple[str, dict]:
        parsed_actions = parse_action_to_structure_output(
            text=model_response,
            factor=1000,
            origin_resized_height=screenshot_height,
            origin_resized_width=screenshot_width,
            model_type="qwen25vl",
        )

        translated_code = parsing_response_to_pyautogui_code(
            responses=parsed_actions,
            image_height=screenshot_height,
            image_width=screenshot_width,
        )

        debug_info = {
            "dpi_settings": {
                "dpi": self._dpi.get_dpi(),
                "scale_factor": self._dpi.get_scale_factor(),
                "awareness_level": self._dpi.get_awareness_name(),
            },
            "monitor_config": {
                "monitors": [
                    {
                        "name": m.name,
                        "resolution": f"{m.width}x{m.height}",
                        "dpi": m.dpi,
                        "scale_factor": m.scale_factor,
                        "is_primary": m.is_primary,
                    }
                    for m in self._monitor_manager.get_config().monitors
                ],
                "virtual_screen": f"{self._monitor_manager.get_config().virtual_screen_width}x{self._monitor_manager.get_config().virtual_screen_height}",
            },
        }

        if self._debug_overlay.is_enabled():
            for action in parsed_actions:
                if "start_box" in action["action_inputs"]:
                    coords = eval(action["action_inputs"]["start_box"])
                    if len(coords) >= 2:
                        x_norm = coords[0]
                        y_norm = coords[1]
                        cursor_pos = self._cursor_tracker.get_current_position()
                        self._debug_overlay.record_transformation(
                            x_norm, y_norm, screenshot_width, screenshot_height,
                            cursor_pos.x, cursor_pos.y,
                        )

        return translated_code, debug_info

    def get_coordinate_mapping(
        self,
        model_x_norm: float,
        model_y_norm: float,
        screenshot_width: int,
        screenshot_height: int,
    ) -> dict:
        screen_coords = self._coordinate_mapper.model_to_screen(
            model_x_norm, model_y_norm, screenshot_width, screenshot_height
        )

        transform_info = self._coordinate_mapper.get_transform_info(
            model_x_norm, model_y_norm, screenshot_width, screenshot_height
        )

        physical_coords = self._coordinate_mapper.denormalize_to_physical(
            model_x_norm, model_y_norm
        )

        return {
            "normalized": (model_x_norm, model_y_norm),
            "screen": screen_coords,
            "physical": physical_coords,
            "monitor": transform_info.monitor_name,
            "scale_factor": transform_info.scale_factor,
        }

    def verify_click_accuracy(self, x: int, y: int) -> dict:
        cursor_pos = self._cursor_tracker.get_current_position()
        is_accurate, error = self._coordinate_mapper.verify_click_target(
            cursor_pos.x, cursor_pos.y, x, y
        )
        return {
            "expected": (x, y),
            "actual": (cursor_pos.x, cursor_pos.y),
            "error_pixels": error,
            "is_accurate": is_accurate,
        }

    def run_calibration(self) -> dict:
        if self._calibrator is None:
            self._calibrator = MouseCalibrator(cursor_tracker=self._cursor_tracker)

        result = self._calibrator.run_calibration()
        self._coordinate_mapper.set_calibration_offset(
            result.offset_x, result.offset_y
        )

        return {
            "offset": (result.offset_x, result.offset_y),
            "confidence": result.confidence,
            "samples": result.sample_count,
        }

    def get_debug_report(self) -> str:
        return self._debug_overlay.create_test_report()


def demo():
    print("=" * 60)
    print("Windows Runtime Demo")
    print("=" * 60)

    agent = WindowsAwareAgent(debug=True)

    print("\nDPI Settings:")
    print(f"  DPI: {agent._dpi.get_dpi()}")
    print(f"  Scale Factor: {agent._dpi.get_scale_factor():.2f}x")
    print(f"  Awareness: {agent._dpi.get_awareness_name()}")

    print("\nMonitor Configuration:")
    config = agent._monitor_manager.get_config()
    for i, monitor in enumerate(config.monitors):
        print(f"  Monitor {i+1}: {monitor.name}")
        print(f"    Resolution: {monitor.width}x{monitor.height}")
        print(f"    DPI: {monitor.dpi}")
        print(f"    Scale: {monitor.scale_factor:.2f}x")
        print(f"    Primary: {monitor.is_primary}")

    print("\nVirtual Screen: {}x{}".format(
        config.virtual_screen_width, config.virtual_screen_height
    ))

    print("\nCoordinate Mapping Demo:")
    for x_norm, y_norm in [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]:
        mapping = agent.get_coordinate_mapping(x_norm, y_norm, 1920, 1080)
        print(f"  ({x_norm:.2f}, {y_norm:.2f}) -> Screen: {mapping['screen']}, Physical: {mapping['physical']}")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo()