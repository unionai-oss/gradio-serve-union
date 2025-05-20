from union import Resources, task, Artifact, FlyteDirectory, current_context, ImageSpec
from transformers import AutoProcessor, AutoModelForVision2Seq
from pathlib import Path
from typing import Annotated

# Create Union Artifact 
SmolVLM = Artifact(name="SmolVLM-Instruct")

# ----------------------------------------------------------------------
# Define the container image
# ----------------------------------------------------------------------
container_image = ImageSpec(
    name="gradio-serve",
    packages=[
        "union==0.1.181",
        "flytekit==1.15.4",
        "transformers==4.51.3",
        "torch==2.5.1", 
        "accelerate==1.6.0",
        "pillow==11.2.1",
    ],
    builder="union",
)

# ----------------------------------------------------------------------
# Download the model
# ----------------------------------------------------------------------
@task(
    container_image=container_image,
    cache=True,
    cache_version="0.0017",  # Update cache version
    requests=Resources(cpu="2", mem="9Gi"),
)
def download_model(
    model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
) -> Annotated[FlyteDirectory, SmolVLM]:

    working_dir = Path(current_context().working_directory)
    saved_model_dir = working_dir / "saved_model"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    model.save_pretrained(saved_model_dir)
    processor.save_pretrained(saved_model_dir)

    return SmolVLM.create_from(saved_model_dir)


# union run --remote 2_cv_images/model.py download_model
