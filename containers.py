from flytekit import ImageSpec, Resources
# from union.actor import ActorEnvironment

container_image = ImageSpec(
     name="gradio-serve",
    requirements="requirements.txt",
    builder="union",
)
