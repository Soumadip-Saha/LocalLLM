import code
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nest_asyncio
import warnings
warnings.filterwarnings("ignore")

class UserInput(BaseModel):
    userId: str = Field("admin", example="admin")
    query: str = Field(..., example="What is the capital of India?")
    chats: List = Field([], example=[{"question": "What is the capital of India?", "answer": "New Delhi"}])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_TEMPLATE = """<s> [INST] \
You are an AI assistant. You will be given some contexts and a question. \
Your task is to answer the question based on the context only. You are not supposed to answer \
the question out of context. If you are unable to answer the question, you can say \
"Sorry, I don't know the answer. Please rephrase the question." \
\n\n Context: {context}\nQuestion: \n {question} [/INST]\
\n Answer: </s>\
"""

chat_template = code.templates.BaseTemplate(CHAT_TEMPLATE, ["context", "question"])

QUESTION_GENERATOR = """<s> [INST] \
    You are given a chat history and a question. Your task is to generate a stand-alone \
    question based on the chat history and the question. \
    \n\nChat History: {chat_history}\nQuestion: {question} [/INST]\
    \nStand-alone Question: </s>\
    """
question_generator = code.templates.BaseTemplate(QUESTION_GENERATOR, ["chat_history", "question"])

query_template = code.templates.BaseTemplate("Instruct: Given a query, retrieve relevant documents that answer the query.\nQuery: {query}", ["query"])

db = code.vector_database.VectorDatabase(
    host="https://54.174.178.103:4100",
    index_name="mcube_genai_v1",
    user_name="elastic",
    password="zDP1wbqb3LBcxh1D=KGt"
)

embedding_model = code.embeddings.MistralEmbeddings("http://54.174.178.103:4000/create_embeddings")

retriever = code.Retriever(
    vector_database=db,
    embedding_model=embedding_model,
    query_template=query_template
)

llm = code.llms.Mixtral(
    model_url="http://54.174.178.103:4000/generate_text"
)

chat_agent = code.agents.RAGChatAgent(
    retriever=retriever,
    llm_model=llm,
    question_generator=question_generator,
    chat_template=chat_template
)

@app.post("/generate_mcube_ans")
async def generate_mcube_ans(user_input: UserInput):
    for message in user_input.chats:
        chat_agent.messages.append({"role": "user", "message": message["question"]})
        chat_agent.messages.append({"role": "assistant", "message": message["answer"]})
    answer = chat_agent.generate(user_input.query)
    chat_agent.messages = []
    return {"answer": answer, "userId": user_input.userId, "query": user_input.query, "references": []}

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)