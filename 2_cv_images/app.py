"""
This serves as a deployment script for the SmolVLM-Instruct model with a Gradio app.
"""
import os
from datetime import timedelta
from union import Artifact, Resources
from union.app import App, Input, ScalingMetric
from flytekit.extras.accelerators import L4
from containers import container_image

# Point to VLM artifact

SmolVLM = Artifact(name="SmolVLM-Instruct")

gradio_app = App(
    name="vlm-gradio",
    inputs=[
        Input(
            name="downloaded-model",
            value=SmolVLM.query(),
            download=True,
        )
    ],
    container_image=container_image,
    port=8080,
    include=["./main.py"],  # Include your gradio app
    args=["python", "main.py"],
    limits=Resources(cpu="2", mem="24Gi", gpu="1", ephemeral_storage="20Gi"),
    requests=Resources(cpu="2", mem="24Gi", gpu="1", ephemeral_storage="20Gi"),
    accelerator=L4,
    min_replicas=0,
    max_replicas=1,
    scaledown_after=timedelta(minutes=10),
    scaling_metric=ScalingMetric.Concurrency(2),
)

# union deploy apps 2_cv_images/app.py vlm-gradio
