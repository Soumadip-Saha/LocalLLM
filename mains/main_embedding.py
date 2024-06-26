from fastapi import FastAPI
from pydantic import BaseModel, Field
import code
from typing import List, Union, Dict
import os
import torch
import uvicorn
from pyngrok import ngrok, conf
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware

class Item(BaseModel):
    text: Union[str, List[str]] = Field(..., example=["What is the capital of India.", "India is a great country. The capital of India is New Delhi."])


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = code.embeddings.EmbeddingModel(device=device)

os.environ["NGROK"] = "2cvxEEs3DFfLJzGqUJCY6L3l2M6_6Pcq1C7D6WX3pkxGi4TzT"
conf.get_default().auth_token = os.environ["NGROK"]

@app.post("/create_embeddings")
async def create_embedding(item: Item):
    if isinstance(item.text, list):
        embeddings = model.get_embedding(item.text)
    else:
        embeddings = model.get_embedding([item.text])
    return {"embeddings": embeddings}


if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
