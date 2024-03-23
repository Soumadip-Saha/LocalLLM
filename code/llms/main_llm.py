from llm_model import download_model
from llama_cpp import Llama
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from pyngrok import ngrok, conf
import nest_asyncio
import contextlib

DEFAULT_MODEL_URL = "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

model_path = download_model(DEFAULT_MODEL_URL, model_dir="/content/Models", model_name="mixtral.gguf")

print("Loading Mixtral model...")
model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=18,
    verbose=False
)
print("Mixtral model loaded successfully.")

class Item(BaseModel):
    prompt: str = Field(..., example="What is the capital of India?")
    max_tokens: int = Field(512, example=512)
    stop: str = Field(["</s>"], example=["</s>"])
    temperature: float = Field(0.7, example=0.7)
    stream: bool = Field(True, example=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate_text")
async def generate_text(item: Item):
    stream = model.create_completion(
        prompt=item.prompt,
        max_tokens=item.max_tokens,
        stop=item.stop,
        temperature=item.temperature,
        echo=False,
        stream=item.stream
    )
    response = {"answer": ""}
    for message in stream:
        response["answer"] += message['choices'][0]['text']
    return response

if __name__ == "__main__":
    os.environ["NGROK"] = "2cvxEEs3DFfLJzGqUJCY6L3l2M6_6Pcq1C7D6WX3pkxGi4TzT"
    conf.get_default().auth_token = os.environ["NGROK"]
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)