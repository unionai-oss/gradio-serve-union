"""
This is a Gradio app for the SmolVLM-Instruct model.
You can upload an image or take photo with webcam and get a vision-language response.
"""

import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load model from Union artifact or fallback local path
try:
    from union_runtime import get_input
    model_path = Path(get_input("downloaded-model"))
except:
    model_path = Path("saved_model")

# Load processor and model
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inference function
def vlm_infer(image: Image.Image) -> str:
    start = time.time()

    # Prompt with <image> placeholder required by IDEFICS3
    prompt = "<|user|>\n<image>\nWhat’s going on in this photo?\n<|end|>\n<|assistant|>"

    # Pass all inputs as keyword arguments to avoid conflict
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)

    # Decode output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    latency = (time.time() - start) * 1000

    return f"{generated_text.strip()}\n\n⚡ {device.type.upper()} | {latency:.1f} ms"

# Gradio UI
demo = gr.Interface(
    fn=vlm_infer,
    inputs=gr.Image(type="pil", label="Upload or Take a Photo"),
    outputs=gr.Text(label="SmolVLM Output"),
    title="SmolVLM-Instruct: Vision-Language Model",
    description="Upload an image to generate a vision-language response using SmolVLM (IDEFICS3).",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
