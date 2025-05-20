from flytekit import ImageSpec

container_image = ImageSpec(
    name="gradio-serve",
    requirements="requirements.txt",
    builder="union",
    cuda="11.8",
)
