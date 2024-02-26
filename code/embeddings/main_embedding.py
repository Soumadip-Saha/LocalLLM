from fastapi import FastAPI
from pydantic import BaseModel, Field
from embedding_model import EmbeddingModel
import uvicorn

class Item(BaseModel):
    text: Union[dict, List[dict]] = Field(..., example={"query":True, "text": "What is the capital of India."})

app = FastAPI()

@app.post("/embed")
async def create_embedding(item: Item):
    if isinstance(item.text, list):
        embeddings = model.get_embedding(item.text)
    else:
        embeddings = model.get_embedding([item.text])
    return {"embeddings": embeddings}


if __name__ == "__main__":
    model = EmbeddingModel()
    uvicorn.run(app, port=8000)