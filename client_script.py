#!/usr/bin/env python3
# Client Script for UI-TARS

import os
import re
import json
import base64
import argparse
from PIL import Image
from io import BytesIO

# Optional dependency - only needed if using OpenAI API format
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI package not installed. Using requests instead.")
    import requests

def add_box_token(input_string):
    """
    Adds box tokens to the model output coordinates.
    This is needed for processing the model's raw output.
    """
    # Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'", 
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'"
                )
            processed_actions.append(updated_action)
        
        # Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string

def encode_image(image_path):
    """Encode an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_model_openai_compatible(base_url, api_key, messages, image_path=None):
    """Query the model using OpenAI-compatible API."""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package is required. Install with 'pip install openai'")
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # If an image is provided, add it to the user's latest message
    if image_path:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                # Add base64 encoded image
                base64_image = encode_image(image_path)
                # Add the image to the content
                content = [{"type": "text", "text": messages[i]["content"]}]
                content.append(
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                )
                messages[i]["content"] = content
                break
    
    # Process assistant messages for proper box token formatting
    for message in messages:
        if message["role"] == "assistant" and isinstance(message["content"], str):
            message["content"] = add_box_token(message["content"])
    
    # Make the API call
    response = client.chat.completions.create(
        model="tgi",  # Model name used by HuggingFace TGI
        messages=messages,
        temperature=0.0,
        max_tokens=400,
        stream=False
    )
    
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='UI-TARS client script for interacting with the model API.')
    parser.add_argument('--api-url', type=str, default=os.environ.get('HF_BASE_URL', ''),
                        help='API endpoint URL')
    parser.add_argument('--api-key', type=str, default=os.environ.get('HF_API_KEY', ''),
                        help='API key for authentication')
    parser.add_argument('--task', type=str, default="Click on the search button",
                        help='Task description for the model')
    parser.add_argument('--image', type=str, default='',
                        help='Path to screenshot image (optional)')
    parser.add_argument('--messages-file', type=str, default='',
                        help='Path to JSON file containing message history (optional)')
    
    args = parser.parse_args()
    
    # Check if API URL and key are provided
    if not args.api_url or not args.api_url.startswith('http'):
        print("Error: Valid API URL is required.")
        print("Set with --api-url or HF_BASE_URL environment variable.")
        return
    
    if not args.api_key:
        print("Error: API key is required.")
        print("Set with --api-key or HF_API_KEY environment variable.")
        return
    
    # Load message history if provided, otherwise create a new one
    if args.messages_file and os.path.exists(args.messages_file):
        with open(args.messages_file, 'r') as f:
            messages = json.load(f)
    else:
        # Default system prompt from prompts.py
        system_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""
        # Create a new message history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.task}
        ]
    
    # Query the model
    try:
        response = query_model_openai_compatible(
            args.api_url, 
            args.api_key, 
            messages,
            args.image
        )
        print("\nModel Response:")
        print(response)
        
        # Add the response to messages and save if message file is provided
        messages.append({"role": "assistant", "content": response})
        if args.messages_file:
            with open(args.messages_file, 'w') as f:
                json.dump(messages, f, indent=2)
    
    except Exception as e:
        print(f"Error querying the model: {e}")

if __name__ == "__main__":
    main()