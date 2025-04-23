# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment
- Language: Python
- Task: UI-TARS is a multimodal agent for GUI interaction

## Commands
- No explicit build/lint/test commands found in the codebase
- For coordinate processing: `python coordinate_processing_script.py`
- For visualization: Use matplotlib to display coordinate outputs

## Code Style
- Indent: 4 spaces
- Quotes: Double quotes for strings
- Imports: Standard library first, then third-party, then local imports
- Error handling: Use specific exceptions with descriptive messages
- Naming: snake_case for functions/variables, UPPER_CASE for constants
- Documentation: Docstrings for functions (as seen in smart_resize)
- Comments: Descriptive comments for complex operations

## Dependencies
- PIL/Pillow for image processing
- matplotlib for visualization
- re for parsing model outputs
- Other common imports: json, math, io

## Model-Specific Notes
- Coordinates are processed with IMAGE_FACTOR=28
- Model outputs need to be rescaled to original dimensions
- Parse model action outputs carefully for accurate coordinate extraction