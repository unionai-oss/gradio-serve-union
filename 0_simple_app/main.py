"""
This is a simple Gradio app that takes a name and an intensity level as input,
and returns a greeting message. The app is designed to be deployed using Union from app.py.
"""

import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)


# union deploy apps 0_simple_app/app.py gradio-app
