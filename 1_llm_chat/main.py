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

# Chat function compatible with Gradio type="messages"
def chat_fn(message, history):
    if history is None:
        history = []

    # Construct prompt from history
    prompt = ""
    for msg in history:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}\n"
    prompt += f"<|user|>\n{message}\n<|assistant|>\n"

    # Tokenize and generate response
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

    # Append to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history

# Define Gradio Chat Interface
chat_interface = gr.ChatInterface(
    fn=chat_fn,
    title="Qwen3 Chatbot",
    textbox=gr.Textbox(placeholder="Ask me anything...", container=True, scale=7),
    multimodal=False,
    theme="default",
    type="messages",
    save_history=True,
)

# Launch app
if __name__ == "__main__":
    chat_interface.launch(server_name="0.0.0.0", server_port=8080)
