import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from union_runtime import get_input

# Load model path from Union artifact input
model_path = get_input("downloaded-model")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
model.eval()

# Chat function using list-of-lists for history (as required by type="tuple")
def chat_fn(message, history):
    if history is None:
        history = []

    # Ensure format is list of lists
    history = [list(pair) for pair in history]

    # Construct prompt
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"<|user|>\n{user_msg}\n<|assistant|>\n{bot_msg}\n"
    prompt += f"<|user|>\n{message}\n<|assistant|>\n"

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("<|assistant|>\n")[-1].strip()

    # Append and return
    history.append([message, response])
    return response

# Define Gradio Chat Interface
chat_interface = gr.ChatInterface(
    fn=chat_fn,
    title="Qwen3 Chatbot",
    chatbot=gr.Chatbot(height=400, type='messages'),
    textbox=gr.Textbox(placeholder="Ask me anything...", container=True, scale=7),
    multimodal=False,
    theme="default",
    type="messages",  # Very important to match return format
)

# Launch app
if __name__ == "__main__":
    chat_interface.launch(server_name="0.0.0.0", server_port=8080)
