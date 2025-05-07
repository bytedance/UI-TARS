# UI-TARS 1.5 HuggingFace Endpoint Deployment Guide

## 1. HuggingFace Inference Endpoints Cloud Deployment

We use HuggingFace's Inference Endpoints platform to quickly deploy a cloud-based model.

### Deployment Steps

1. **Access the Deployment Interface**  
    - Click [Deploy from Hugging Face](https://endpoints.huggingface.co/catalog)  
    ![Deploy from Hugging Face](https://huggingface.co/datasets/JjjFangg/Demo_video/resolve/main/deployment_1_formal.png?download=true)  
    - Select the model `UI-TARS 1.5 7B` and click **Import Model**  
    ![Import Model](https://huggingface.co/datasets/JjjFangg/Demo_video/resolve/main/deployment_2_formal.png?download=true)  

2. **Configure Settings**
    - **Hardware Configuration**  
        - In the `Hardware Configuration` section, choose a GPU instance. Here are the recommendations for different model sizes:  
            - For the 7B model, select `GPU L40S 1GPU 48G` (Recommended: Nvidia L4 / Nvidia A100).  
        ![Hardware Configuration](https://huggingface.co/datasets/JjjFangg/Demo_video/resolve/main/deployment_3_formal.png?download=true)

    - **Container Configuration**  
        - Set the following parameters:  
            - `Max Input Length (per Query)`: 65536
            - `Max Batch Prefill Tokens`: 65536
            - `Max Number of Tokens (per Query)`: 65537
        ![Container Configuration](https://huggingface.co/datasets/JjjFangg/Demo_video/resolve/main/deployment_4_formal.png?download=true)

    - **Environment Variables**  
        - Add the following environment variables:  
            - `CUDA_GRAPHS=0` to avoid deployment failures. For details, refer to [issue 2875](https://github.com/huggingface/text-generation-inference/issues/2875).  
            - `PAYLOAD_LIMIT=8000000` to prevent request failures due to large images. For details, refer to [issue 1802](https://github.com/huggingface/text-generation-inference/issues/1802).  
        ![Environment Variables](https://huggingface.co/datasets/JjjFangg/Demo_video/resolve/main/deployment_5_formal.png?download=true)

    - **Create Endpoint**  
        - Click **Create** to set up the endpoint.  
        ![Create Endpoint](https://huggingface.co/datasets/JjjFangg/Demo_video/resolve/main/deployment_6_formal.png?download=true)

    - **Enter Setup**  
        - Once the deployment is finished, you will see the confirmation page and need to enter the settings page.  
        ![Complete](https://huggingface.co/datasets/JjjFangg/Demo_video/resolve/main/deployment_7_formal.png?download=true)
    
    - **Update URI** -
        - Go to the Container page, set the Container URI to ghcr.io/huggingface/text-generation-inference:3.2.1, and **click Update Endpoint to apply the changes**. 
        ![Complete](https://huggingface.co/datasets/JjjFangg/Demo_video/resolve/main/deployment_8_formal.png?download=true)


## 2. API Usage Example

### **Python Test Code**  
```python
# pip install openai
import io
import re
import json
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI

def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string

client = OpenAI(
    base_url="https:xxx",
    api_key="hf_xxx"
)

result = {}
messages = json.load(open("./data/test_messages.json"))
for message in messages:
    if message["role"] == "assistant":
        message["content"] = add_box_token(message["content"])
        print(message["content"])

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=messages,
    top_p=None,
    temperature=0.0,
    max_tokens=400,
    stream=True,
    seed=None,
    stop=None,
    frequency_penalty=None,
    presence_penalty=None
)

response = ""
for message in chat_completion:
    response += message.choices[0].delta.content
print(response)
```

### **Expected Output** ###
```python
Thought: 我看到Preferences窗口已经打开了，但这里显示的都是系统资源相关的设置。要设置图片的颜色模式，我得先看看左侧的选项列表。嗯，"Color Management"这个选项看起来很有希望，应该就是处理颜色管理的地方。让我点击它看看里面有什么选项。
Action: click(start_box='(177,549)')
```
