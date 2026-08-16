import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_tars.action_parser import (
    parsing_response_to_pyautogui_code,
    parse_action,
    parse_action_to_structure_output,
)


class TestActionParser(unittest.TestCase):
    def test_parse_action(self):
        action_str = "click(point='<point>200 300</point>')"
        result = parse_action(action_str)
        self.assertEqual(result['function'], 'click')
        self.assertEqual(result['args']['point'], '<point>200 300</point>')

    def test_parse_action_to_structure_output(self):
        text = "Thought: test\nAction: click(point='<point>200 300</point>')"
        actions = parse_action_to_structure_output(
            text,
            factor=1000,
            origin_resized_height=224,
            origin_resized_width=224,
            min_pixels=1,
        )
        self.assertEqual(actions[0]['action_type'], 'click')
        self.assertIn('start_box', actions[0]['action_inputs'])

    def test_parse_action_with_box_tokens(self):
        text = (
            "Thought: test\n"
            "Action: click(start_box='<|box_start|>(112,224)<|box_end|>')"
        )
        actions = parse_action_to_structure_output(
            text,
            factor=1000,
            origin_resized_height=224,
            origin_resized_width=224,
            min_pixels=1,
        )
        self.assertEqual(actions[0]["action_type"], "click")
        self.assertEqual(
            actions[0]["action_inputs"]["start_box"],
            "[0.5, 1.0, 0.5, 1.0]",
        )

    def test_parse_doubao_seed_compact_action(self):
        text = (
            "<think_never_used_51bce0c785ca2f68081bfa7d91973934>"
            "Click the orientation menu."
            "</think_never_used_51bce0c785ca2f68081bfa7d91973934>"
            "click>point>point>112 56"
        )
        actions = parse_action_to_structure_output(
            text,
            factor=1000,
            origin_resized_height=224,
            origin_resized_width=224,
            min_pixels=1,
        )
        self.assertEqual(actions[0]["thought"], "Click the orientation menu.")
        self.assertEqual(actions[0]["action_type"], "click")
        self.assertEqual(
            actions[0]["action_inputs"]["start_box"],
            "[0.5, 0.25, 0.5, 0.25]",
        )

    def test_parsing_response_to_pyautogui_code(self):
        responses = {"action_type": "hotkey", "action_inputs": {"hotkey": "ctrl v"}}
        code = parsing_response_to_pyautogui_code(responses, 224, 224)
        self.assertIn('pyautogui.hotkey', code)

    def test_scroll_with_end_box_generates_drag_gesture(self):
        responses = {
            "action_type": "scroll",
            "action_inputs": {
                "start_box": "[0.8, 0.2, 0.8, 0.2]",
                "end_box": "[0.2, 0.8, 0.2, 0.8]",
            },
        }
        code = parsing_response_to_pyautogui_code(responses, 1000, 1000)
        self.assertIn("pyautogui.moveTo(800.0, 200.0)", code)
        self.assertIn("pyautogui.dragTo(200.0, 800.0, duration=1.0)", code)

    def test_drag_accepts_two_point_tuples(self):
        responses = {
            "action_type": "drag",
            "action_inputs": {
                "start_box": ("897", "1208"),
                "end_box": ("244", "1208"),
            },
        }
        code = parsing_response_to_pyautogui_code(responses, 1, 1)
        self.assertIn("pyautogui.moveTo(897.0, 1208.0)", code)
        self.assertIn("pyautogui.dragTo(244.0, 1208.0, duration=1.0)", code)


if __name__ == '__main__':
    unittest.main()
