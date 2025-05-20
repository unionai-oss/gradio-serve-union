
"""
# Simple Gradio App Deployment Example
"""
from datetime import timedelta
from union import Resources
from union.app import App, ScalingMetric
from containers import container_image

gradio_app = App(
    name="gradio-app",
    container_image=container_image, # image that contains the environment and dependencies needed to run the app
    port=8080, # The port on which the app will be served
    include=["./main.py"],  # Include your gradio code
    args=["python", "main.py"], # Command to run your app inside the container
    limits=Resources(cpu="2", mem="8Gi"), # Maximum resources allocated (CPU, memory, GPU) — hard limit
    requests=Resources(cpu="2", mem="8Gi"), # Minimum resources requested from the scheduler — soft requirement
    min_replicas=0, # Minimum number of instances (pods) running — allows scale-to-zero when idle
    max_replicas=1, # Maximum number of instances — restricts auto-scaling to 1 replica
    scaledown_after=timedelta(minutes=5), # Time to wait before scaling down when traffic is low
    scaling_metric=ScalingMetric.Concurrency(2), # Auto-scaling based on concurrent user requests; 2 concurrent users per replica
    # requires_auth=False # Uncomment to make app public.
)

# union deploy apps 0_simple_app/app.py gradio-app
