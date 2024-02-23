import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
from torch import Tensor

DEFAULT_TASK = """Given a query, retrieve relevant documents that answer the query."""

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
    def __init__(self, model_dir:str, max_length=4096, task:str=DEFAULT_TASK, device:str="cuda"):
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir)
        self.model=AutoModel.from_pretrained(model_dir).to(device)
        self.task=task
        self.max_length=max_length-1

    def get_embedding(self, texts:List[str], **kwargs):
        for idx, text in enumerate(texts):
            texts[idx] = f"Instruct {self.task}\nQuery: {text}\n"
        
        batch_dict = self.tokenizer(
            texts, 
            max_length=self.max_length,
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            return_tensors=kwargs.get("return_tensors", "pt")
        )
        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()