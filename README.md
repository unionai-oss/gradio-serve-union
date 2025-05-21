# Gradio Examples on Union.ai

<a target="_blank" href="https://colab.research.google.com/github/unionai-oss/gradio-serve-union/blob/main/tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The easiest way to follow this tutorial is to [run it in Google Colab](https://colab.research.google.com/github/unionai-oss/gradio-serve-union/blob/main/tutorial.ipynb). You can do this by clicking the "Open In Colab" button above. This will open the tutorial in a new tab, where you can run the code cells interactively.

If you want to run this example on your local machine, you can do so by running the following command:

```bash
git clone https://github.com/unionai-oss/gradio-serve-union/
cd gradio-serve-union
# (Create an environment if needed)
pip install -r requirements.txt
```

### Hello World Gradio App
```
# 👇 Run this command to deploy the application
union deploy apps 0_simple_app/app.py gradio-app
```
The command above is using files from the [`0_simple_app/`](0_simple_app//app.py) folder that got cloned on setup.


### Qwen3 Chat Applications
👇 Run this task to download the model files for the LLM chat example.
You only need to run this once. Then you can see the model in the artifact tab on Union.ai
```
union run --remote 1_llm_chat/model.py download_model
```
👇 Run this command to deploy the chatbot app with the model downloaded above
note: You need to wait for the model to be downloaded before running this command

```
union deploy apps 1_llm_chat/app.py gradio-chat
```

### SmolVLM Applications

👇 Run this task to download the model files for the VLM chat example.
You only need to run this once. Then you can see the model in the artifact tab on Union.ai

```
union run --remote 2_cv_images/model.py download_model
```

👇 Run this command to deploy the Gradio app with the model downloaded above
note: You need to wait for the model to be downloaded before running this command

```
union deploy apps 2_cv_images/app.py vlm-gradio
```