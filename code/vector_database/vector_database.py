import requests
from elasticsearch import Elasticsearch
import code
from elasticsearch.helpers import bulk
from typing import List, Dict

DEFAULT_INDEX_MAPPING = "code\\vector_database\INDEX.json"

class VectorDatabase:
    def __init__(self, host, index_name, user_name, password):
        self.es = Elasticsearch(host, basic_auth=(user_name, password), verify_certs=False)
        self.index_name = index_name
    
    def create_index(self, mapping=None):
        if mapping is None:
            with open(DEFAULT_INDEX_MAPPING, "r") as index_file:
                mapping = index_file.read().strip()
        
        if self.es.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} already exists.")
            if input("Do you want to delete it? (y/n): ").lower() == "y":
                self.es.indices.delete(index=self.index_name)
            else:
                return False

        print(f"Creating index {self.index_name}")
        _ = self.es.indices.create(index=self.index_name, body=mapping)
        return True
    
    def insert(self, document):
        _ = self.es.index(index=self.index_name, document=document)
    
    def bulk_insert(self, documents):
        success, info = bulk(self.es, documents, index=self.index_name)
        if success:
            return True
        else:
            return info
    
    def add_documents(
        self, documents: List[Dict], embedding_model: code.embeddings.MistralEmbeddings, batch_size: int = 10
    ):
        inserted_docs = 0
        for idx in range(0, len(documents), batch_size):
            batch = documents[idx:idx+batch_size]
            embeddings = embedding_model.get_embeddings([doc["content"] for doc in batch])["embeddings"]
            for doc, embedding in zip(batch, embeddings):
                doc["embedding"] = embedding
            if self.bulk_insert(batch) == True:
                inserted_docs += len(batch)
            else:
                print("Error inserting documents.")
        print(f"Inserted {inserted_docs} documents.")
        return "Done"
        
    def similarity_search(self, query_vector: List[float], top_k=5):
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }

        response = self.es.search(
            index=self.index_name, 
            body={
                "size": top_k,
                "query": script_query,
                "_source": ["content", "metadata.source"],
            }
        )

        documents = [
            {
                "content": hit["_source"]["content"],
                "source": hit["_source"]["metadata"]["source"],
                "score": hit["_score"]
            }
            for hit in response["hits"]["hits"]
        ]

        return documents
