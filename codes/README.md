# ui-tars

A python package for parsing VLM-generated GUI action instructions into executable pyautogui codes.

---

## Introduction

`ui-tars` is a Python package for parsing VLM-generated GUI action instructions, automatically generating pyautogui scripts, and supporting coordinate conversion and smart image resizing.

- Supports multiple VLM output formats (e.g., Qwen-VL, Seed-VL)
- Automatically handles coordinate scaling and format conversion
- One-click generation of pyautogui automation scripts

---

## Quick Start

### Installation

```bash
pip install ui-tars
# or
uv pip install ui-tars
```

### Parse output into structured actions

```python
from ui_tars.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code

response = "Thought: Click the button\nAction: click(point='<point>200 300</point>')"
original_image_width, original_image_height = 1920, 1080
parsed_dict = parse_action_to_structure_output(
    response,
    factor=1000,
    origin_resized_height=original_image_height,
    origin_resized_width=original_image_width,
    model_type="doubao"
)
print(parsed_dict)
parsed_pyautogui_code = parsing_response_to_pyautogui_code(
    responses=parsed_dict,
    image_height=original_image_height,
    image_width=original_image_width
)
print(parsed_pyautogui_code)
```

### Generate pyautogui automation script

```python
from ui_tars.action_parser import parsing_response_to_pyautogui_code

pyautogui_code = parsing_response_to_pyautogui_code(parsed_dict, original_image_height, original_image_width)
print(pyautogui_code)
```

### Visualize coordinates on the image (optional)

```python
from PIL import Image, ImageDraw
import ast
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("your_image_path.png")
start_box = parsed_dict[0]["action_inputs"]["start_box"]
coordinates = ast.literal_eval(start_box)
x1 = int(coordinates[0] * original_image_width)
y1 = int(coordinates[1] * original_image_height)
draw = ImageDraw.Draw(image)
radius = 5
draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill="red", outline="red")
plt.imshow(np.array(image))
plt.axis("off")
plt.show()
```

---

## API Documentation

### parse_action_to_structure_output

```python
def parse_action_to_structure_output(
    text: str,
    factor: int,
    origin_resized_height: int,
    origin_resized_width: int,
    model_type: str = "qwen25vl",
    max_pixels: int = 16384 * 28 * 28,
    min_pixels: int = 100 * 28 * 28
) -> list[dict]:
    ...
```

**Description:**
Parses output action instructions into structured dictionaries, automatically handling coordinate scaling and box/point format conversion.

Supported coordinate/action forms include:

```text
Action: click(point='<point>200 300</point>')
Action: click(start_box='(200,300)')
Action: click(start_box='<|box_start|>(200,300)<|box_end|>')
Action: scroll(start_box='(800,200)', end_box='(200,800)')
<think_...>reasoning</think_...>click>point>point>200 300
```

For Qwen2.5-VL style models, coordinates are treated as absolute coordinates
in the resized image and are normalized by `smart_resize`. For older
relative-coordinate models, pass the appropriate `factor`.

**Parameters:**
- `text`: The output string
- `factor`: Scaling factor
- `origin_resized_height`/`origin_resized_width`: Original image height/width
- `model_type`: Model type (e.g., "qwen25vl", "doubao")
- `max_pixels`/`min_pixels`: Image pixel upper/lower limits

**Returns:**
A list of structured actions, each as a dict with fields like `action_type`, `action_inputs`, `thought`, etc.

---

### parsing_response_to_pyautogui_code

```python
def parsing_response_to_pyautogui_code(
    responses: dict | list[dict],
    image_height: int,
    image_width: int,
    input_swap: bool = True
) -> str:
    ...
```

**Description:**
Converts structured actions into a pyautogui script string, supporting click, type, hotkey, drag, scroll, and more.

`scroll` supports both mouse-wheel style actions with `direction` and
gesture-style actions with `start_box` + `end_box`. Gesture-style scrolls are
emitted as a `moveTo` + `dragTo` sequence, matching how mobile/emulator scrolls
are usually executed.

**Parameters:**
- `responses`: Structured actions (dict or list of dicts)
- `image_height`/`image_width`: Image height/width
- `input_swap`: Whether to use clipboard paste for typing (default True)

**Returns:**
A pyautogui script string, ready for automation execution.

---

## Contribution

Contributions, issues, and suggestions are welcome!

---

## License

Apache-2.0 License
