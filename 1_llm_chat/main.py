from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from union_runtime import get_input
import threading
from transformers import TextIteratorStreamer

# --------------------------
# Load model path from Union artifact input
# --------------------------
model_path = get_input("downloaded-model")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
model.eval()


def chat_fn(message, history):
    """
    Function to handle chat messages and generate responses.
    """

    messages = [{"role": "user", "content": message}]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": 2048,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.3,
            "eos_token_id": tokenizer.eos_token_id,
        },
    )
    thread.start()

    thinking_prefix = "🤔 **Thinking:**\n"
    answer_prefix = "\n\n🧠 **Answer:**\n"

    current_section = "thinking"
    yielded_thinking = ""
    yielded_answer = ""

    for token in streamer:
        if current_section == "thinking":
            yielded_thinking += token
            if "</think>" in yielded_thinking:
                # Split at </think> and switch to answer phase
                thinking_text, remainder = yielded_thinking.split("</think>", 1)
                yield thinking_prefix + thinking_text.strip()
                current_section = "answer"
                yielded_answer += remainder
                yield answer_prefix + yielded_answer.strip()
            else:
                yield thinking_prefix + yielded_thinking.strip()
        else:
            yielded_answer += token
            yield answer_prefix + yielded_answer.strip()


# --------------------------
# Define Gradio interface
# --------------------------
chat_interface = gr.ChatInterface(
    fn=chat_fn,
    title="Qwen3 Chatbot",
    textbox=gr.Textbox(placeholder="Ask me anything...", container=True, scale=7),
    multimodal=False,
    theme="default",
    type="messages",
)

# --------------------------
# Launch Gradio app
# --------------------------
if __name__ == "__main__":
    chat_interface.launch(server_name="0.0.0.0", server_port=8080)

# union deploy apps 1_llm_chat/app.py gradio-chat

