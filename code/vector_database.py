import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class VectorDatabase:
    def __init__(self, host, index_name, user_name, password):
        self.es = Elasticsearch(host, basic_auth=(user_name, password), verify_certs=False)
        self.index_name = index_name
    
    def create_index(self, mapping=None):
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
