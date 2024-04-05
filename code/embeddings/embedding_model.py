import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from typing import List, Dict, Union
from torch import Tensor
import requests

DEFAULT_TASK = """Given a query, retrieve relevant documents that answer the query."""
DEFAULT_MODEL = "Salesforce/SFR-Embedding-Mistral"


def download_model(url, model_dir="/content/Models", model_name="model.gguf"):
    # TO-DO: Implement github repository copy    
    pass

def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class EmbeddingModel():
    def __init__(self, model_dir:str = None, max_length=4096, device:str="auto", bnb_config=None):
        if model_dir is None:
            print("No model directory provided. Using Salesforce's Mistral model.")
            model_dir = DEFAULT_MODEL
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir)
            
        self.model=AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            resume_download=True
        )
        self.max_length=max_length-1

    def get_embedding(self, texts:List[str], **kwargs):
        """
        texts: List of dictionaries with the following structure:
        [
            "What is the capital of India.",
            "India is a great country. The capital of India is New Delhi."
        ]
        """
        
        batch_dict = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            return_tensors=kwargs.get("return_tensors", "pt")
        )
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

class MistralEmbeddings():
    def __init__(self, model_url: str):
        self.model_url = model_url
    
    def get_embeddings(self, texts: Union[str, List[str]]):
        embeddings = requests.post(self.model_url, json={"text": texts})
        return embeddings.json()