from code.templates import BaseTemplate, get_chat_history
from code.vector_database import VectorDatabase
from code.templates import BaseTemplate, get_chat_history
from code.llms import Mixtral
from code.embeddings import MistralEmbeddings
from code.templates import BaseTemplate
import requests
from typing import List, Dict
from code.utils import Retriever

# TODO: Complete the implementation of the Retriever class


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
    
    def get_answer(self, query: str, docs: List[Dict], **kwargs):
        prompt = self.chat_template.get_prompt({"question": query, "context": "\n\n".join([doc["content"] for doc in docs])})
        answer = self.llm.get_response(prompt)
        return answer

    def generate(self, query):
        self.messages.append({"role": "user", "message": query})
        chat_history = get_chat_history(self.messages[-11:])
        stand_alone_query = self.get_standalone_query(query, chat_history)
        print("Stand-alone Query: ", stand_alone_query, end="\n\n")
        docs = self.retriever.get_docs(stand_alone_query)
        retrieved_docs = "\n".join([doc["content"] for doc in docs])
        print(f"Retrieved Documents: {retrieved_docs}\n\n")
        answer = self.get_answer(query, docs)
        self.messages.append({"role": "assistant", "message": answer})
        return answer
    
    