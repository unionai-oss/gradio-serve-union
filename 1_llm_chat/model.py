from union import Resources, task, Artifact, FlyteDirectory, current_context, ImageSpec
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Annotated

# Create Union Artifact 
Qwen3Model8b = Artifact(name="qwen3-model")

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
        "accelerate==1.6.0"
    ],
    builder="union",
)

# ----------------------------------------------------------------------
# Download the model
# ----------------------------------------------------------------------

@task(
    container_image=container_image,
    cache=True,
    cache_version="0.0005",
    requests=Resources(cpu="2", mem="9Gi"),
)
def download_model(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> Annotated[FlyteDirectory, Qwen3Model8b]:

    working_dir = Path(current_context().working_directory)
    saved_model_dir = working_dir / "saved_model"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(saved_model_dir)
    tokenizer.save_pretrained(saved_model_dir)

    # return FlyteDirectory(saved_model_dir)
    return Qwen3Model8b.create_from(saved_model_dir)

# union run --remote 1_llm_chat/model.py download_model
