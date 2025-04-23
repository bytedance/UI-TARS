FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install UI-TARS dependencies
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    accelerate==0.23.0 \
    bitsandbytes==0.41.1 \
    pillow==10.0.1 \
    matplotlib==3.7.3 \
    numpy==1.24.3 \
    sentencepiece==0.1.99 \
    openai==1.0.0 \
    requests==2.31.0 \
    pydantic==2.5.1 \
    safetensors==0.4.0 \
    scipy==1.11.3 \
    vllm==0.6.1

# Copy project files
COPY . /app/

# Create directories for model and data
RUN mkdir -p /app/model /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_MODEL_ID="ByteDance-Seed/UI-TARS-1.5-7B"
ENV HF_HOME="/app/model"
ENV TRANSFORMERS_CACHE="/app/model"

# Download UI-TARS model from Hugging Face (comment out if you want to download separately)
RUN echo "Starting model download..." && \
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    print('Downloading tokenizer...'); \
    tokenizer = AutoTokenizer.from_pretrained('${HF_MODEL_ID}', trust_remote_code=True); \
    print('Tokenizer downloaded successfully'); \
    # If you have enough memory and want to download the model directly, uncomment the next line \
    # model = AutoModelForCausalLM.from_pretrained('${HF_MODEL_ID}', trust_remote_code=True, device_map='auto'); \
    # print('Model downloaded successfully');" || echo "Model will be downloaded at runtime"

# Create model server script
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import torch\n\
from vllm import LLM, SamplingParams\n\
from vllm.entrypoints.openai.api_server import serve_vllm_api_server\n\
from transformers import AutoTokenizer\n\
\n\
def main():\n\
    model_id = os.environ.get("HF_MODEL_ID", "ByteDance-Seed/UI-TARS-1.5-7B")\n\
    print(f"Starting server with model: {model_id}")\n\
    \n\
    # Load tokenizer\n\
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n\
    \n\
    # Start vLLM server\n\
    serve_vllm_api_server(\n\
        model=model_id,\n\
        tensor_parallel_size=1,  # Change based on available GPUs\n\
        gpu_memory_utilization=0.9,\n\
        trust_remote_code=True,\n\
        dtype="bfloat16",  # Use float16 if bfloat16 is not supported\n\
        host="0.0.0.0",\n\
        port=8000\n\
    )\n\
\n\
if __name__ == "__main__":\n\
    main()\n\
' > /app/server.py && chmod +x /app/server.py

# Create inference script
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
import json\n\
import base64\n\
import argparse\n\
from PIL import Image\n\
from io import BytesIO\n\
import requests\n\
\n\
def encode_image(image_path):\n\
    with open(image_path, "rb") as image_file:\n\
        return base64.b64encode(image_file.read()).decode("utf-8")\n\
\n\
def query_model(image_path, instruction, server_url="http://localhost:8000/v1/chat/completions"):\n\
    # Encode the image\n\
    base64_image = encode_image(image_path)\n\
    \n\
    # Prepare the messages with system prompt from prompts.py\n\
    with open("/app/prompts.py", "r") as f:\n\
        prompts_content = f.read()\n\
    \n\
    # Extract computer use prompt\n\
    import re\n\
    computer_prompt = re.search(r\'COMPUTER_USE = \"\"\"(.+?)\"\"\"\', prompts_content, re.DOTALL)\n\
    if computer_prompt:\n\
        system_prompt = computer_prompt.group(1).replace("{language}", "English").replace("{instruction}", instruction)\n\
    else:\n\
        system_prompt = "You are a GUI agent. You are given a task and your action history, with screenshots."\n\
    \n\
    # Prepare the API request\n\
    headers = {"Content-Type": "application/json"}\n\
    payload = {\n\
        "model": "UI-TARS",\n\
        "messages": [\n\
            {"role": "system", "content": system_prompt},\n\
            {"role": "user", "content": [\n\
                {"type": "text", "text": instruction},\n\
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}\n\
            ]}\n\
        ],\n\
        "temperature": 0.01,\n\
        "max_tokens": 512\n\
    }\n\
    \n\
    # Make the API request\n\
    try:\n\
        response = requests.post(server_url, headers=headers, json=payload)\n\
        response.raise_for_status()\n\
        result = response.json()\n\
        return result["choices"][0]["message"]["content"]\n\
    except Exception as e:\n\
        return f"Error: {str(e)}"\n\
\n\
def main():\n\
    parser = argparse.ArgumentParser(description="UI-TARS Model Inference")\n\
    parser.add_argument("--image", required=True, help="Path to screenshot image")\n\
    parser.add_argument("--instruction", required=True, help="Task instruction")\n\
    parser.add_argument("--server", default="http://localhost:8000/v1/chat/completions", help="Server URL")\n\
    \n\
    args = parser.parse_args()\n\
    \n\
    result = query_model(args.image, args.instruction, args.server)\n\
    print(result)\n\
\n\
if __name__ == "__main__":\n\
    main()\n\
' > /app/inference.py && chmod +x /app/inference.py

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "serve" ]; then\n\
    echo "Starting UI-TARS server..."\n\
    python /app/server.py\n\
elif [ "$1" = "infer" ]; then\n\
    echo "Running inference..."\n\
    python /app/inference.py --image "$2" --instruction "$3"\n\
elif [ "$1" = "process-coordinates" ]; then\n\
    echo "Processing coordinates..."\n\
    python /app/coordinate_processing_script.py --image "$2" --model-output "$3" --output "$4"\n\
elif [ "$1" = "analyze-webpage" ]; then\n\
    echo "Analyzing webpage..."\n\
    python /app/webpage_analyzer.py --image "$2" ${3:+--output "$3"}\n\
else\n\
    echo "UI-TARS Docker container"\n\
    echo "Usage:"\n\
    echo "  serve                   - Start the model server"\n\
    echo "  infer IMAGE INSTRUCTION - Run inference on an image"\n\
    echo "  process-coordinates IMAGE MODEL_OUTPUT OUTPUT - Process and visualize coordinates"\n\
    echo "  analyze-webpage IMAGE [OUTPUT_FILE] - Analyze a webpage screenshot and output description"\n\
    echo "Environment:"\n\
    echo "  HF_MODEL_ID - HuggingFace model ID (default: ByteDance-Seed/UI-TARS-1.5-7B)"\n\
fi\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["help"]