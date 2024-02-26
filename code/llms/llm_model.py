from llama_cpp import Llama
import os
import requests
from tqdm import tqdm


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
    return model_path
    

def load_model(**kwargs):
    return Llama(**kwargs)
