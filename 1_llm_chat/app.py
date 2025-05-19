import os
from datetime import timedelta
from union import Artifact, ImageSpec, Resources
from union.app import App, Input, ScalingMetric
from flytekit.extras.accelerators import GPUAccelerator, L4

# Point to your object detection model artifact
Qwen3Model8b = Artifact(name="qwen3-model")

image_spec = ImageSpec(
    name="gradio-chat",
    packages=[
        "gradio==5.29.0",
        "torch==2.5.1",
        "union-runtime>=0.1.18",
        "transformers==4.51.3",
        "accelerate==1.6.0",
    ],
    cuda="11.8",
    builder="union",
)

gradio_app = App(
    name="gradio-chat",
    inputs=[
        Input(
            name="downloaded-model",
            value=Qwen3Model8b.query(),
            download=True,
        )
    ],
    container_image=image_spec,
    port=8080,
    include=["./main.py"],  # Include your gradio app
    args=["python", "main.py"],
    limits=Resources(cpu="2", mem="8Gi", gpu="1"),
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    accelerator=L4,
    min_replicas=0,
    max_replicas=1,
    scaledown_after=timedelta(minutes=5),
    scaling_metric=ScalingMetric.Concurrency(2),
)


# union deploy apps 1_llm_chat/app.py gradio-chat
