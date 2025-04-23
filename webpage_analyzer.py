#!/usr/bin/env python3
# UI-TARS Webpage Analysis Script

import os
import sys
import json
import base64
import argparse
from PIL import Image
from io import BytesIO
import requests

def encode_image(image_path):
    """Encode an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_webpage(image_path, server_url="http://localhost:8000/v1/chat/completions"):
    """
    Analyze a webpage screenshot and return a detailed plaintext description.
    Uses the UI-TARS model to perform the analysis.
    """
    # Encode the image
    base64_image = encode_image(image_path)
    
    # Create analysis instruction
    instruction = """
    Analyze this webpage screenshot and provide a detailed plaintext description of:
    1. Page layout and structure
    2. Navigation elements
    3. Main content sections
    4. Interactive elements (buttons, forms, menus)
    5. Visual design elements

    Focus on being comprehensive but concise. Organize your description logically.
    """
    
    # Prepare the API request
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "UI-TARS",
        "messages": [
            {"role": "system", "content": "You are an expert UI analyzer. Given a screenshot of a webpage, you provide detailed and structured descriptions of the interface elements, layout, and content organization."},
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    # Make the API request
    try:
        response = requests.post(server_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}\n\nIf the UI-TARS server is not running, start it with:\ndocker-compose up -d ui-tars"

def main():
    parser = argparse.ArgumentParser(description="UI-TARS Webpage Analysis")
    parser.add_argument("--image", required=True, help="Path to webpage screenshot")
    parser.add_argument("--server", default="http://localhost:8000/v1/chat/completions", help="Server URL")
    parser.add_argument("--output", help="Output file path (optional, outputs to console if not specified)")
    
    args = parser.parse_args()
    
    print(f"Analyzing webpage screenshot: {args.image}")
    result = analyze_webpage(args.image, args.server)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Analysis saved to: {args.output}")
    else:
        print("\n=== WEBPAGE ANALYSIS ===\n")
        print(result)

if __name__ == "__main__":
    main()