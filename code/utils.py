from .vector_database import VectorDatabase
from .embeddings import MistralEmbeddings
from .templates import BaseTemplate

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
