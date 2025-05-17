
import os
from datetime import timedelta
from union import Artifact, ImageSpec, Resources
from union.app import App, Input, ScalingMetric
from flytekit.extras.accelerators import GPUAccelerator, L4

# Point to your object detection model artifact
# FRCCNFineTunedModel = Artifact(name="frccn_fine_tuned_model")

image_spec = ImageSpec(
    name="gradio-app",
    packages=[
        "gradio==5.29.0",
        "union-runtime>=0.1.18",
    ],
    builder="union",
)

gradio_app = App(
    name="gradio-app",
    container_image=image_spec,
    port=8080,
    include=["./main.py"],  # Include your gradio code
    args=["python", "main.py"],
    limits=Resources(cpu="2", mem="8Gi"),
    requests=Resources(cpu="2", mem="8Gi"),
    min_replicas=0,
    max_replicas=1,
    scaledown_after=timedelta(minutes=2),
    scaling_metric=ScalingMetric.Concurrency(2),
)

# union deploy apps app.py gradio-app
