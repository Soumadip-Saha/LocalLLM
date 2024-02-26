from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Union
from code.embeddings import EmbeddingModel
import os

app = FastAPI()

class Item(BaseModel):
    text: Union[dict, List[dict]] = Field(..., example={"query":True, "text": "What is the capital of India."})

class Documents(BaseModel):
    documents: Union[dict, List[dict]] = Field(..., example={"content": "This is a sample document"})

class Query(BaseModel):
    query_vector: List[float] = Field(..., example=[0.1, 0.2, 0.3])

# model = EmbeddingModel(model_dir="path_to_your_model")
# database = VectorDatabase(os.environ["ES_HOST"], os.environ["ES_INDEX"], os.environ["ES_USER"], os.environ["ES_PASSWORD"])

@app.post("/embed")
async def create_embedding(item: Item):
    if isinstance(item.text, list):
        embedding = model.get_embedding(item.text)
    else:
        embedding = model.get_embedding([item.text])
    return {"embedding": embedding}

@app.post("/bulk_insert")
async def bulk_insert_documents(documents: Documents):
    response = db.bulk_insert(documents=documents.documents)
    if response is True:
        return {"status": "success"}
    else:
        return {"status": "failure", "error": response}

@app.post("/similarity_search")
async def similarity_search(query: Query):
    results = db.similarity_search(query.query_vector)
    return {"results": results}

