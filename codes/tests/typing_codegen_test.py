# SPDX-License-Identifier: Apache-2.0
import os
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_tars.action_parser import parsing_response_to_pyautogui_code


class TestTypingCodegen(unittest.TestCase):
    def assert_typed(self, content, expected, enter=False):
        for input_swap in (False, True):
            with self.subTest(content=content, input_swap=input_swap):
                gui = ModuleType("pyautogui")
                gui.write = Mock()
                gui.hotkey = Mock()
                gui.press = Mock()
                clipboard = ModuleType("pyperclip")
                clipboard.copy = Mock()
                clock = ModuleType("time")
                clock.sleep = Mock()
                code = parsing_response_to_pyautogui_code(
                    {"action_type": "type", "action_inputs": {"content": content}},
                    224,
                    224,
                    input_swap=input_swap,
                )
                with patch.dict(
                    sys.modules,
                    {
                        "pyautogui": gui,
                        "pyperclip": clipboard,
                        "time": clock,
                    },
                ):
                    exec(compile(code, "<generated-typing>", "exec"), {})  # noqa: S102 - controlled code, offline GUI stubs
                if input_swap:
                    clipboard.copy.assert_called_once_with(expected)
                    gui.hotkey.assert_called_once_with("ctrl", "v")
                    gui.write.assert_not_called()
                else:
                    gui.write.assert_called_once_with(expected, interval=0.1)
                    clipboard.copy.assert_not_called()
                if enter:
                    gui.press.assert_called_once_with("enter")
                else:
                    gui.press.assert_not_called()

    def test_actual_terminal_newline(self):
        self.assert_typed("python\n", "python", enter=True)

    def test_windows_path(self):
        self.assert_typed(r"C:\new\report.txt", r"C:\new\report.txt")

    def test_embedded_newline(self):
        self.assert_typed("first\nsecond", "first\nsecond")

    def test_apostrophe_and_backslash(self):
        self.assert_typed("an author's \\notes", "an author's \\notes")

    def test_trailing_backslash(self):
        self.assert_typed("folder\\", "folder\\")

    def test_ordinary_text(self):
        self.assert_typed("plain text", "plain text")

    def test_unicode(self):
        self.assert_typed("中文 🐋", "中文 🐋")


if __name__ == "__main__":
    unittest.main()
