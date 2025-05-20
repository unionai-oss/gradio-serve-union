"""
This task downloads the SmolVLM-Instruct model from Hugging Face and saves it to a specified directory.
You can adjust the model to another from Hugging Face VLM by changing the model_name parameter.
"""

from union import Resources, task, Artifact, FlyteDirectory, current_context, ImageSpec
from transformers import AutoProcessor, AutoModelForVision2Seq
from pathlib import Path
from typing import Annotated
from containers import container_image

# Create Union Artifact 
SmolVLM = Artifact(name="SmolVLM-Instruct")


# ----------------------------------------------------------------------
# Download the model
# ----------------------------------------------------------------------
@task(
    container_image=container_image,
    cache=True,
    cache_version="1.0",
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
