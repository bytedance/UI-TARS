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
            text, factor=1000, origin_resized_height=224, origin_resized_width=224
        )
        self.assertEqual(actions[0]['action_type'], 'click')
        self.assertIn('start_box', actions[0]['action_inputs'])

    def test_parsing_response_to_pyautogui_code(self):
        responses = {"action_type": "hotkey", "action_inputs": {"hotkey": "ctrl v"}}
        code = parsing_response_to_pyautogui_code(responses, 224, 224)
        self.assertIn('pyautogui.hotkey', code)

    def test_scroll_left_and_right_emit_hscroll(self):
        right = {
            "action_type": "scroll",
            "action_inputs": {
                "direction": "right",
                "start_box": "[0.5, 0.25, 0.5, 0.25]",
            },
        }
        code = parsing_response_to_pyautogui_code(
            right, image_height=1000, image_width=2000
        )
        self.assertIn("pyautogui.hscroll(5, x=1000.0, y=250.0)", code)

        left = {
            "action_type": "scroll",
            "action_inputs": {"direction": "Left"},
        }
        code = parsing_response_to_pyautogui_code(
            left, image_height=1000, image_width=2000
        )
        self.assertIn("pyautogui.hscroll(-5)", code)
        self.assertNotIn("pyautogui.scroll(", code)

        up = {
            "action_type": "scroll",
            "action_inputs": {"direction": "up"},
        }
        code = parsing_response_to_pyautogui_code(up, image_height=100, image_width=100)
        self.assertIn("pyautogui.scroll(5)", code)
        self.assertNotIn("hscroll", code)


if __name__ == '__main__':
    unittest.main()
