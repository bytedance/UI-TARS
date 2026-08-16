# UI-TARS evaluation and reproduction notes

This document collects the public reproduction details that are spread across
the README, prompt templates, deployment guide, and issue discussions. It is
intended as a checklist for local evaluation scripts. It does not replace the
official benchmark harnesses or disclose private evaluation infrastructure.

## Environment

The Python package lives under `codes/` and declares its runtime metadata in
`codes/pyproject.toml`. There is no separate `requirements.txt`; use the
project metadata instead:

```bash
cd codes
uv sync
make test
```

For lightweight parser-only checks without `uv`, the standard unittest runner
also works after any optional test dependencies have been installed:

```bash
cd codes
python -m unittest discover tests '*_test.py'
```

`tests/inference_test.py` uses Pillow, so direct full test discovery requires
the optional image stack from the project environment.

## Prompt templates

The prompt templates used by the public package are in
`codes/ui_tars/prompt.py`:

- `COMPUTER_USE_DOUBAO`: desktop GUI tasks such as OSWorld-style computer use.
- `MOBILE_USE_DOUBAO`: mobile GUI tasks and Android-emulator tasks.
- `GROUNDING_DOUBAO`: single-step grounding tasks where only `Action:` is
  expected.

For browser benchmarks such as WebVoyager and Online-Mind2Web, UI-TARS is a
vision-language GUI agent: the model should receive screenshots/visual input,
not only HTML or accessibility-tree text. Use the computer/browser task prompt
style and keep the action grammar consistent with the parser.

## Inference settings

For deterministic benchmark reproduction, use greedy decoding:

```text
temperature = 0
top_p = 1
```

For Qwen2.5-VL based local inference, keep the image pixel budget aligned with
the coordinate-processing utilities. The default maximum in this repository is:

```text
max_pixels = 16384 * 28 * 28
```

When comparing scores, verify that your request stack, model endpoint, image
resizing, prompt, action parser, history window, and benchmark environment all
match. Small changes in any of those pieces can change grounding coordinates
and long-horizon task success rates.

## Conversation/history format

For multi-step GUI tasks, include the current screenshot and the recent action
history in the same style used by the target benchmark harness. A common
pattern is:

- current user instruction
- recent screenshots or observations
- previous `Thought:` / `Action:` assistant turns
- current screenshot before requesting the next action

Do not mix incompatible action grammars in the same run. For example, if the
prompt asks for `click(start_box='...')`, the postprocessor should also expect
`start_box`. If the prompt asks for `click(point='<point>x y</point>')`, run it
through the coordinate conversion path before executing the action.

## Coordinate handling

UI-TARS has used two coordinate conventions:

- Qwen2.5-VL style outputs absolute coordinates in the resized image.
- Older relative-coordinate flows use a fixed scale factor such as `1000`.

Use `parse_action_to_structure_output` from `ui_tars.action_parser` so those
formats are normalized before execution.

Supported input forms include:

```text
Action: click(point='<point>200 300</point>')
Action: click(start_box='(200,300)')
Action: click(start_box='<|box_start|>(200,300)<|box_end|>')
Action: scroll(start_box='(800,200)', end_box='(200,800)')
```

For very large screenshots, make sure the dimensions passed to the parser are
the same dimensions used during model preprocessing. Passing the original
monitor resolution when the model saw a resized image is a common cause of
wrong x/y coordinates.

## Action mapping

The public issue discussion describes the following dataset-level action
normalizations:

| Dataset/source | Normalization |
| --- | --- |
| GUI-Odyssey | `incomplete` -> `call_user`; `KEY_BACK` -> `press_back`; `KEY_HOME` -> `press_home`; `KEY_APPSELECT` -> `press_appselect` |
| Android | `navigate_back` -> `press_back`; `navigate_home` -> `press_home` |
| Mind2Web | `enter` -> `press_enter` |

For Mind2Web `select` examples, the public clarification is that the coordinate
represents the center point of the expanded option box to be selected. Value
selection is commonly executed by clicking/selecting that target point, and
long dropdowns may require scroll actions before the selection is visible.

## Mobile scroll vs drag

Mobile/emulator environments often execute both scroll and drag as pointer
gestures. Use:

```text
scroll(start_box='(x1,y1)', end_box='(x2,y2)')
drag(start_box='(x1,y1)', end_box='(x2,y2)')
```

when the executor expects touch-like movement from a start point to an end
point. Use `scroll(point='...', direction='up|down|left|right')` only when the
executor maps that direction to a platform-specific wheel or swipe operation.

If the model's natural-language thought says one direction but the coordinates
describe another, the executor will follow the coordinates. Debug the parsed
`start_box` and `end_box` first.

## Benchmark-specific notes

### OSWorld

Use the OSWorld UI-TARS agent script referenced by the main README for the
closest public reproduction path. Ensure your local fork of OSWorld, prompt
template, history window, image preprocessing, and model endpoint match the
reported setup before comparing leaderboard numbers.

### Android World

Android World is not just OSWorld with different screenshots. Use the mobile
prompt/action space, preserve the expected Android action names, and verify
that the app launcher/home-screen state is consistent across runs. Prompt
changes or device UI changes can materially change scores.

### WebVoyager and Online-Mind2Web

Run them as multimodal browser-use tasks. The model should process the
screenshot/visual state and output a GUI action. If your harness uses DOM-only
or accessibility-tree-only inputs, it is not evaluating the same capability.

### ScreenSpot and ScreenSpot-Pro

For grounding-only evaluation, use the grounding prompt, greedy decoding, and
the same image pixel budget passed to the model. Normalize model outputs
through the action parser before scoring points.

## Deployment endpoint checks

For OpenAI-compatible deployments, the base URL usually needs the `/v1` suffix,
for example:

```python
client = OpenAI(
    base_url="https://your-endpoint.example.com/v1",
    api_key="...",
)
```

If a desktop or benchmark client receives HTTP 404 from `/chat/completions`,
first confirm both the base URL and model name expected by your serving backend.
