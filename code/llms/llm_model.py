from llama_cpp import Llama
import os
import requests
from tqdm import tqdm
from typing import Optional, List


def download_model(url, model_dir="/content/Models", model_name="model.gguf"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Failed to download the model. HTTP status code: {response.status_code}")
        if os.path.exists(model_path):
            os.remove(model_path)
    return model_path


# class Mixtral:
#     # write description
#     """
#     This is a LLM model wrapper for Mixtral model. It calls the hosted Mixtral model to generate responses.
#     model_url: str - URL of the Mixtral model
#     bos: str - Beginning of sentence token. Default is "<s>"
#     eos: str - End of sentence token. Default is "</s>"
#     instruct_start_token: str - Instruction start token. Default is "[INST]"
#     instruct_end_token: str - Instruction end token. Default is "[/INST]"
#     system_template: str - System template for the model
#     """
#     def __init__(
#         self, 
#         model_url: str, system_template: str, bos: str = "<s>", eos: str = "</s>",
#         instruct_start_token: str = "[INST]", instruct_end_token: str = "[/INST]"
#     ):
#         self.model_url = model_url
#         self.system_template = system_template
#         self.bos = bos
#         self.eos = eos
#         self.instruct_start_token = instruct_start_token
#         self.instruct_end_token = instruct_end_token
#         self.user_messages = []
#         self.assistant_messages = []

#     def get_response(self, prompt: str):
#         self.user_messages.append(prompt)
#         prompt = self.format_prompt()
#         response = requests.post(self.model_url, json={"prompt": prompt})
#         return response.json()["answer"]
    
#     def format_prompt(self):
#         prompt = f"{self.bos}{self.instruct_start_token} {self.system_template}\n"
#         for idx in range(len(self.user_messages)+len(self.assistant_messages)):
#             if idx % 2 == 0:
#                 prompt += f"User: {user_messages[idx//2]} {self.instruct_end_token}\n"
#             else:
#                 prompt += f"Assistant: {assistant_messages[idx//2]} {self.eos}\n"
#         return prompt

class Mixtral:
    def __init__(self, model_url: str):
        self.model_url = model_url
    
    def get_response(self, prompt: str):
        response = requests.post(self.model_url, json={"prompt": prompt})
        return response.json()["answer"]