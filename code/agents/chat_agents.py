


class retriever:
    def __init__(self, vector_database):
        self.vector_db = vector_database
        
    def get_docs(self):
        # TO-DO: Implement a function to retrieve documents from the database
        raise NotImplementedError("This function has not been implemented yet.")
    
    def get_embeddings(self):
        # TO-DO: Implement a function to get embeddings from the embeddings model
        raise NotImplementedError("This function has not been implemented yet.")
    



class chatbot:
    def __init__(self, system_template: str, retriever, llm_model):
        self.retriever = retriever
        self.llm = llm_model
        self.messages = {"user": [], "assistant": []}
        self.system_template = system_template
        # self.
    
    def get_standalone_query(self, query):
        return f"Instruct {self.llm.task}\nQuery: {query}\n"
    
    def generate(self, query):
        # standalone_query = self.
        raise NotImplementedError("This function has not been implemented yet.")
    
    