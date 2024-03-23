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
    
def process_messages(messages, bos_token, eos_token):
    result = ""
    for index, message in enumerate(messages):
        role = message['role']
        content = message['content']

        if (role == 'user') != (index % 2 == 0):
            raise ValueError('Conversation roles must alternate user/assistant/user/assistant/...')

        if role == 'user':
            formatted_message = f"[INST] {content} [/INST]"
        elif role == 'assistant':
            formatted_message = f"{content}{eos_token}"
        else:
            raise ValueError('Only user and assistant roles are supported!')

        result += formatted_message

    return f"{bos_token}{result}"

messages = [
    {'role': 'user', 'content': 'Hello!'},
    {'role': 'assistant', 'content': 'Hi there!'},
    {'role': 'user', 'content': 'How are you?'},
    {'role': 'assistant', 'content': 'I am fine, thank you!'},
    {'role': 'user', 'content': 'What is the Capital of India?'}
]

class Mixtral:
    """
    """
    def __init__(
        self, 
        model_url: str, system_template: str, bos: str = "<s>", eos: str = "</s>",
        instruct_start_token: str = "[INST]", instruct_end_token: str = "[/INST]"
    ):
        self.model_url = model_url
        self.system_template = system_template
        self.bos = bos
        self.eos = eos
        self.instruct_start_token = instruct_start_token
        self.instruct_end_token = instruct_end_token
        self.user_messages = []
        self.assistant_messages = []

    def get_response(self, prompt: str):
        return prompt
    
    def format_prompt(self, user_messages: List[str], assistant_messages: List[str]):
        prompt = f"{self.bos}{self.instruct_start_token} {self.system_template}\n"
        for idx in range(len(user_messages)+len(assistant_messages)):
            if idx % 2 == 0:
                prompt += f"User: {user_messages[idx//2]} {self.instruct_end_token}\n"
            else:
                prompt += f"Assistant: {assistant_messages[idx//2]} {self.eos}\n"
        self.user_messages += user_messages
        self.assistant_messages += assistant_messages
        return prompt