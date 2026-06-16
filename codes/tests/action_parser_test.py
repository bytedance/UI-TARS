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

    def test_type_with_trailing_newline_preserves_content(self):
        # After parsing, a real trailing newline becomes the literal "\\n".
        # The trailing newline must be stripped (so Enter is pressed
        # separately) WITHOUT eating real trailing "n"/"\\" characters from
        # the typed text. Regression test for the rstrip("\\n") char-set bug
        # that turned "Login\\n" into "Logi" and "Run\\n" into "Ru".
        for raw, expected in [
            ("Login\\n", "Login"),
            ("Run\\n", "Run"),
            ("python\\n", "python"),
            ("hello world\\n", "hello world"),
        ]:
            responses = {
                "action_type": "type",
                "action_inputs": {"content": raw},
            }
            code = parsing_response_to_pyautogui_code(
                responses, 1080, 1920, input_swap=False
            )
            self.assertIn(f"pyautogui.write('{expected}'", code)
            # A trailing newline still triggers a separate Enter press.
            self.assertIn("pyautogui.press('enter')", code)

    def test_type_without_newline_is_unchanged(self):
        responses = {
            "action_type": "type",
            "action_inputs": {"content": "Login"},
        }
        code = parsing_response_to_pyautogui_code(
            responses, 1080, 1920, input_swap=False
        )
        self.assertIn("pyautogui.write('Login'", code)
        self.assertNotIn("pyautogui.press('enter')", code)


if __name__ == '__main__':
    unittest.main()
