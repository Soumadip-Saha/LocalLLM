from fastapi import FastAPI
from pydantic import BaseModel, Field
from embedding_model import EmbeddingModel
from typing import List, Union, Dict
import torch
import uvicorn
from pyngrok import ngrok
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware

class TextItem(BaseModel):
    query: int = Field(..., example=1)
    text: str = Field(..., example="What is the capital of India.")

class Item(BaseModel):
    items: List[TextItem]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EmbeddingModel(device=device)

@app.post("/embed")
async def create_embedding(item: Item):
    embeddings = model.get_embedding(item.items)
    return {"embeddings": embeddings}


if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
