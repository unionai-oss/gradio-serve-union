import os
from datetime import timedelta
from union import Artifact, Resources
from union.app import App, Input, ScalingMetric
from flytekit.extras.accelerators import L4
from containers import container_image

# Point to your object detection model artifact
Qwen3Model8b = Artifact(name="qwen3-model")

# 
gradio_app = App(
    name="gradio-chat",
    inputs=[
        Input(
            name="downloaded-model",
            value=Qwen3Model8b.query(),
            download=True,
        )
    ],
    container_image=container_image, # image that contains the environment and dependencies needed to run the app
    port=8080, # The port on which the app will be served
    include=["./main.py"],  # Include your gradio app code
    args=["python", "main.py"], # Command to run your app inside the container
    limits=Resources(cpu="2", mem="16Gi", gpu="1"),  # Maximum resources allocated (CPU, memory, GPU) — hard limit
    requests=Resources(cpu="2", mem="16Gi", gpu="1"), # Minimum resources requested from the scheduler — soft requirement
    accelerator=L4,  # Specifies the GPU type to use (e.g., NVIDIA L4 accelerator)
    min_replicas=0, # Minimum number of instances (pods) running — allows scale-to-zero when idle
    max_replicas=1, # Maximum number of instances — restricts auto-scaling to 1 replica
    scaledown_after=timedelta(minutes=5), # Time to wait before scaling down when traffic is low
    scaling_metric=ScalingMetric.Concurrency(2), # Auto-scaling based on concurrent user requests; 2 concurrent users per replica
    # requires_auth=False # Uncomment to make app public.
)


# union deploy apps 1_llm_chat/app.py gradio-chat
