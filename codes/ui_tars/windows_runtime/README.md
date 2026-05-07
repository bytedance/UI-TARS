# Windows Compatibility Layer

A comprehensive Windows compatibility layer for UI-TARS that addresses coordinate translation, DPI scaling, and multi-monitor challenges.

## Features

- **DPI Awareness**: Proper handling of Windows DPI scaling (100%-400%)
- **Per-Monitor DPI**: Support for mixed DPI setups with different scaling per monitor
- **Multi-Monitor Support**: Automatic detection and coordinate translation across multiple monitors
- **Coordinate Normalization**: Convert between normalized (0-1), screen, and physical coordinates
- **Mouse Calibration**: Self-calibrating mouse system for improved click accuracy
- **Cursor Tracking**: Real-time cursor position monitoring and drift detection
- **Debug Overlay**: Comprehensive debugging and testing tools

## Architecture

```
windows_runtime/
├── __init__.py          # Main exports
├── dpi.py               # DPI awareness and scaling
├── monitor_manager.py   # Multi-monitor detection and management
├── coordinate_mapper.py # Coordinate transformation pipeline
├── cursor_tracker.py    # Cursor position tracking
├── calibration.py       # Mouse calibration system
├── overlay_debugger.py  # Debug and testing tools
└── resolution_normalizer.py  # Resolution normalization
```

## Quick Start

```python
from ui_tars.windows_runtime import CoordinateMapper, DPIAwareness, MonitorManager

# Initialize the Windows runtime components
dpi = DPIAwareness()
monitor_manager = MonitorManager()
mapper = CoordinateMapper(dpi_awareness=dpi, monitor_manager=monitor_manager)

# Transform normalized model coordinates to screen coordinates
x_screen, y_screen = mapper.model_to_screen(
    x_norm=0.5,           # Normalized X (0-1)
    y_norm=0.5,           # Normalized Y (0-1)
    screenshot_width=1920,
    screenshot_height=1080
)
```

## Use with UI-TARS

```python
from ui_tars.action_parser import parse_action_to_structure_output
from ui_tars.windows_runtime import WindowsAwareAgent

# Create a Windows-aware agent
agent = WindowsAwareAgent(debug=True)

# Process model response with proper coordinate translation
model_response = "Thought: Click the button\nAction: click(start_box='(500,300)')"
code, debug_info = agent.process_model_response(
    model_response,
    screenshot_width=1920,
    screenshot_height=1080
)
```

## DPI Scaling

Windows uses different coordinate systems depending on DPI settings:

- **Logical coordinates**: The coordinate space used by applications
- **Physical coordinates**: Actual pixel positions on screen

The DPI layer handles conversion between these spaces:

```python
from ui_tars.windows_runtime import DPIAwareness

dpi = DPIAwareness()

# Convert logical to physical
physical_x, physical_y = dpi.logical_to_physical(1000, 500)

# Convert physical to logical
logical_x, logical_y = dpi.physical_to_logical(1500, 750)
```

## Multi-Monitor Support

Detect and handle multiple monitors with different configurations:

```python
from ui_tars.windows_runtime import MonitorManager

manager = MonitorManager()
config = manager.get_config()

for monitor in config.monitors:
    print(f"Monitor: {monitor.name}")
    print(f"  Resolution: {monitor.width}x{monitor.height}")
    print(f"  DPI: {monitor.dpi}")
    print(f"  Position: ({monitor.x}, {monitor.y})")
    print(f"  Scale: {monitor.scale_factor:.2f}x")
```

## Mouse Calibration

For improved click accuracy, run the calibration routine:

```python
from ui_tars.windows_runtime import MouseCalibrator, CursorTracker

calibrator = MouseCalibrator(cursor_tracker=CursorTracker())
result = calibrator.run_calibration()

print(f"Calibration complete!")
print(f"  Offset: ({result.offset_x:.2f}, {result.offset_y:.2f})")
print(f"  Confidence: {result.confidence:.1%}")
```

## Debugging

Enable the debug overlay to see detailed coordinate transformations:

```python
from ui_tars.windows_runtime import DebugOverlay

debug = DebugOverlay()
debug.enable()

# Record transformations
info = debug.record_transformation(
    x_norm=0.5, y_norm=0.5,
    screenshot_width=1920, screenshot_height=1080,
    cursor_x=960, cursor_y=540
)

# Get statistics
stats = debug.get_statistics()
print(f"Accuracy: {stats['success_rate']:.1%}")

# Generate test report
print(debug.create_test_report())
```

## Testing

Run the test suite:

```bash
python -m unittest tests.windows_runtime_test
```

## Platform Support

- Windows 10/11 with Python 3.10+
- Designed for per-monitor DPI awareness
- Falls back gracefully on non-Windows platforms