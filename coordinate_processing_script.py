#!/usr/bin/env python3
# Coordinate Processing Script for UI-TARS

import os
import re
import math
from PIL import Image
import matplotlib.pyplot as plt
import json
import argparse

# Constants from the README_coordinates.md
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def process_coordinates(image_path, model_output, output_path=None):
    """Process coordinates from model output and visualize them on the image."""
    # Extract coordinates using regex
    coordinates_match = re.search(r"start_box='?\((\d+),\s*(\d+)\)'?", model_output)
    if not coordinates_match:
        print("No coordinates found in the model output.")
        return
    
    model_output_width = int(coordinates_match.group(1))
    model_output_height = int(coordinates_match.group(2))
    
    # Open the image
    img = Image.open(image_path)
    width, height = img.size
    print(f'Original image dimensions: {width}x{height}')
    
    # Calculate the new dimensions
    new_height, new_width = smart_resize(height, width)
    new_coordinate = (
        int(model_output_width/new_width * width), 
        int(model_output_height/new_height * height)
    )
    print(f'Resized dimensions: {new_width}x{new_height}')
    print(f'Original model coordinates: ({model_output_width},{model_output_height})')
    print(f'Mapped screen coordinates: {new_coordinate}')
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.scatter([new_coordinate[0]], [new_coordinate[1]], c='red', s=100)  # Mark the point with a red dot
    plt.title('Visualized Coordinate')
    plt.axis('off')  # Set to 'off' to hide the axes
    
    if output_path:
        plt.savefig(output_path, dpi=350)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process UI-TARS model coordinates.')
    parser.add_argument('--image', type=str, default='./data/coordinate_process_image.png',
                        help='Path to the image')
    parser.add_argument('--model-output', type=str, 
                        default="Action: click(start_box='(197,525)')",
                        help='Model output containing coordinates')
    parser.add_argument('--output', type=str, default='./data/processed_image.png',
                        help='Output path for visualization')
    
    args = parser.parse_args()
    
    process_coordinates(args.image, args.model_output, args.output)

if __name__ == "__main__":
    main()