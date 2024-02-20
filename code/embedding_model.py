import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
from torch import Tensor

DEFAULT_TASK = """Given a query, retrieve relevant documents that answer the query."""

def download_model(url, model_dir="/content/Models", model_name="model.gguf"):
    # TO-DO: Implement github repository copy    
    pass

class embedding_model():
    def __init__(self, model_dir:str, max_length=4096, task:str=DEFAULT_TASK, device:str="cuda"):
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir)
        self.model=AutoModel.from_pretrained(model_dir).to(device)
        self.task=task
        self.max_length=max_length-1

    def _format_input(self, texts:List[str]):
        # To-Do: Implement the model to format the input
        for idx, text in enumerate(texts):
            texts[idx] = f"Instruct {self.task}\nQuery: {text}\n"
        return texts

    def _last_token_pool(
        self,
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

    def get_embedding(self, texts:List[str], **kwargs):
        # To-Do: Implement the model to get the embeddings
        texts = self._format_input(texts)
        batch_dict = self.tokenizer(
            texts, 
            max_length=self.max_length,
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            return_tensors=kwargs.get("return_tensors", "pt")
        )
        outputs = self.model(**batch_dict)
        embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


        