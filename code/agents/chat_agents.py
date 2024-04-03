from code.templates import BaseTemplate, get_chat_history
from code.vector_database import VectorDatabase
from code.templates import BaseTemplate, get_chat_history
from code.llms import Mixtral
from code.embeddings import MistralEmbeddings
from code.templates import BaseTemplate
import requests


# TODO: Complete the implementation of the Retriever class
class Retriever:
    def __init__(self, vector_database: VectorDatabase, embedding_model: MistralEmbeddings, query_template: BaseTemplate):
        self.vector_db = vector_database
        self.embedding_model = embedding_model
        self.query_template = query_template
        
    def get_docs(self, query: str, top_k=5):
        query = self.query_template.get_prompt({"query": query})
        query_embedding = self.embedding_model.get_embeddings(query)["embeddings"][0]
        docs = self.vector_db.similarity_search(query_embedding, top_k)
        return docs

class RAGChatAgent:
    def __init__(
        self, retriever: Retriever, llm_model: Mixtral,
        question_generator: BaseTemplate, chat_template: BaseTemplate
    ):
        self.retriever = retriever
        self.llm = llm_model
        self.messages = []
        self.question_generator = question_generator
        self.chat_template = chat_template
    
    def get_standalone_query(self, query: str, chat_history: str):
        prompt = self.question_generator.get_prompt({"chat_history": chat_history, "question": query})
        stand_alone_query = self.llm.get_response(prompt)
        return stand_alone_query
    
    def get_answer(self, query: str, docs: List[dict]):
        prompt = self.chat_template.get_prompt({"query": query, "context": "\n\n".join([doc["content"] for doc in docs])})
        answer = self.llm.get_response(prompt)
        return answer

    def generate(self, query):
        self.messages.append({"role": "user", "message": query})
        chat_history = get_chat_history(self.messages)
        stand_alone_query = self.get_standalone_query(query, chat_history)
        docs = self.retriever.get_docs(stand_alone_query)
        answer = self.get_answer(query, docs)
        self.messages.append({"role": "assistant", "message": answer})
        return answer
    
    