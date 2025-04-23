# Docker Setup for UI-TARS

This guide explains how to build and run UI-TARS inside a Docker container. The setup includes the full UI-TARS-1.5-7B model, allowing you to run inference on images and process coordinates.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with at least 24GB VRAM
- NVIDIA Container Toolkit (nvidia-docker2)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bytedance/UI-TARS.git
cd UI-TARS
```

### 2. Build and Start the Docker Container

```bash
# Build and start the container
docker-compose up -d ui-tars

# Check logs to monitor download progress
docker-compose logs -f ui-tars
```

This will:
- Build the Docker image
- Download the UI-TARS-1.5-7B model from Hugging Face (first run)
- Start a vLLM server exposing an OpenAI-compatible API on port 8000

### 3. Usage

#### Running Inference

Place your screenshots in the `data` directory, then:

```bash
# Using the docker container directly
docker-compose exec ui-tars /app/entrypoint.sh infer /app/data/your_screenshot.png "Click on the search button"

# Or using the inference service
docker-compose --profile infer run ui-tars-infer infer /app/data/your_screenshot.png "Click on the search button"
```

#### Processing Coordinates

```bash
docker-compose exec ui-tars /app/entrypoint.sh process-coordinates \
  /app/data/your_screenshot.png \
  "Action: click(start_box='(197,525)')" \
  /app/data/result.png
```

#### Webpage Analysis

Analyze a webpage screenshot and convert it to a detailed plaintext description:

```bash
# Generate analysis and print to console
docker-compose exec ui-tars /app/entrypoint.sh analyze-webpage /app/data/webpage_screenshot.png

# Save analysis to a file
docker-compose exec ui-tars /app/entrypoint.sh analyze-webpage /app/data/webpage_screenshot.png /app/data/analysis.txt
```

This will produce a structured description of the webpage including:
- Page layout and structure
- Navigation elements
- Main content sections
- Interactive elements (buttons, forms, menus)
- Visual design elements

## Advanced Configuration

### Using a Different Model

You can set the `HF_MODEL_ID` environment variable to use a different model:

```bash
HF_MODEL_ID=ByteDance-Seed/UI-TARS-1.5-7B docker-compose up -d ui-tars
```

### GPU Configuration

Edit the `docker-compose.yml` file to change GPU settings:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # Change based on available GPUs
          capabilities: [gpu]
```

Edit the `server.py` script for tensor parallelism settings:

```python
serve_vllm_api_server(
    model=model_id,
    tensor_parallel_size=1,  # Increase for multi-GPU setup
    # ...
)
```

## Troubleshooting

### CUDA/GPU Issues

If encountering GPU or CUDA errors:

1. Verify NVIDIA drivers are correctly installed:
   ```bash
   nvidia-smi
   ```

2. Check NVIDIA Container Toolkit is installed:
   ```bash
   dpkg -l | grep nvidia-container-toolkit
   ```

3. Try with `float16` instead of `bfloat16` in `server.py` if your GPU doesn't support bfloat16

### Memory Issues

- For lower memory GPUs (16GB), reduce `gpu_memory_utilization` in `server.py`
- Consider CPU-only inference by removing GPU-specific settings and adding:
  ```python
  serve_vllm_api_server(
      model=model_id,
      device="cpu",
      # ...
  )
  ```

## API Documentation

Once the server is running, an OpenAI-compatible API is exposed at:
```
http://localhost:8000/v1/chat/completions
```

Example curl request:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "UI-TARS",
    "messages": [
      {"role": "system", "content": "You are a GUI agent..."},
      {"role": "user", "content": [
        {"type": "text", "text": "Click on the search button"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]}
    ],
    "temperature": 0.01,
    "max_tokens": 512
  }'
```