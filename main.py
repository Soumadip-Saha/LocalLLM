# from typing import List, Union
# from code.vector_database import VectorDatabase
# import os
# from code.splitters import RecursiveCharacterTextSplitter
# from code.loaders import TextFileLoader
# import requests
# import datetime

# # app = FastAPI()

# # class Item(BaseModel):
# #     text: Union[dict, List[dict]] = Field(..., example={"query":True, "text": "What is the capital of India."})

# # class Documents(BaseModel):
# #     documents: Union[dict, List[dict]] = Field(..., example={"content": "This is a sample document"})

# # class Query(BaseModel):
# #     query_vector: List[float] = Field(..., example=[0.1, 0.2, 0.3])

# # model = EmbeddingModel(model_dir="path_to_your_model")



# # @app.post("/bulk_insert")
# # async def bulk_insert_documents(documents: Documents):
# #     response = db.bulk_insert(documents=documents.documents)
# #     if response is True:
# #         return {"status": "success"}
# #     else:
# #         return {"status": "failure", "error": response}

# # @app.post("/similarity_search")
# # async def similarity_search(query: Query):
# #     results = db.similarity_search(query.query_vector)
# #     return {"results": results}

# # if __name__ == "__main__":
# #     loader = TextFileLoader(path=r"D:\Files\LocalLLM\demo_documents\demo.txt", is_folder=False)
# #     docs = loader.load()
# #     print(docs[0]["content"])
# #     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30, length_function=len)
# #     splits = splitter.split_documents(docs)

# #     for idx, split in enumerate(splits, start=1):
# #         print(f"Split {idx}:\n{split}\n\n-------------------------\n\n")

# def read_files(folder_path: str) -> List[dict]:
#     files = TextFileLoader(path=folder_path, is_folder=True)
#     files = files.load()
#     return files

# def split_documents(documents: List[dict]) -> List[dict]:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30, length_function=len)
#     splits = splitter.split_documents(documents)
#     return splits

# def get_embeddings(texts: List[dict], queries: List[bool], embedding_model_api) -> List[dict]:
#     inputs = [{"query":is_query, "text": text["content"]} for is_query, text in zip(queries, texts)]
#     response = requests.post(embedding_model_api, json={"text": inputs})
#     embeddings = response.json()
#     outputs = []
#     for idx in range(len(texts)):
#         if not inputs[idx]["query"]:
#             outputs.append({"content": texts[idx]["content"], "embedding": embeddings["embeddings"][idx], "metadata": texts[idx]["metadata"]})
#         else:
#             outputs.append({"content": texts[idx]["content"], "embedding": embeddings["embeddings"][idx]})
#     return outputs

# def add_to_database(documents: List[dict], database: VectorDatabase):
#     for doc in documents:
#         doc["metadata"]["creation_date"] = datetime.datetime.now().isoformat()
#     return database.bulk_insert(documents)

# def main():
#     embedding_model_api = "https://7ee5-104-199-149-72.ngrok-free.app/embed"
#     vector_database_host = "https://localhost:9200/"
#     vector_database_index = "test_index"
#     vector_database_user = "elastic"
#     vector_database_password = "xiayY08ILG7iiYuf3Xx5"
#     database = VectorDatabase(host=vector_database_host, index_name=vector_database_index, user_name=vector_database_user, password=vector_database_password)

#     # folder_path = "./demo_documents"
#     # files = read_files(folder_path)
#     # splits = split_documents(files)
#     # embeddings = get_embeddings(splits, [False]*len(splits), embedding_model_api)
#     # response = add_to_database(embeddings, database)
#     # print(response)
#     while True:
#         question = input("Enter a query: ")
#         query_embedding = get_embeddings(texts=[{"content": question}], queries=[True], embedding_model_api=embedding_model_api)
#         results = database.similarity_search(query_embedding[0]["embedding"], top_k=1)
#         print(results[0])

# if __name__ == "__main__":
#     main()

import code

CHAT_TEMPLATE = """<s> [INST] \
    You are an AI assistant. You will be given some contexts with a chat history and a question. \
    Your task is to answer the question based on the context and the chat history. You are not supposed to answer \
    the question out of context or chat history. If you are unable to answer the question, you can say \
    "Sorry, I don't know the answer. Please rephrase the question." \
    \n\n Context: {context}\nChat History: {chat_history}\nQuestion: \n {question} [/INST]\
    \n Answer: </s>\
    """

chat_template = code.templates.BaseTemplate(CHAT_TEMPLATE, ["context", "chat_history", "question"])

QUESTION_GENERATOR = """<s> [INST] \
    You are given a chat history and a question. Your task is to generate a stand-alone \
    question based on the chat history and the question. \
    \n\nChat History: {chat_history}\nQuestion: {question} [/INST]\
    \nStand-alone Question: </s>\
    """
question_generator = code.templates.BaseTemplate(QUESTION_GENERATOR, ["chat_history", "question"])

db = code.vector_database.VectorDatabase(
    host="host-goes-here",
    index_name="index-name-goes-here",
    user_name="user-name-goes-here",
    password="password-goes-here"
)

query_embedder = code.embeddings.MistralEmbeddings(
    model_url="model-url-goes-here"
)

retriever = code.Retriever(
    vector_database=db,
    embedding_model=query_embedder
)

llm = code.llms.Mixtral(
    model_url="model-url-goes-here"
)

chat_agent = code.agents.RAGChatAgent(
    retriever=retriever,
    llm_model=llm,
    question_generator=question_generator,
    chat_template=chat_template
)

if __name__ == "__main__":
    while True:
        query = input("Enter a query: ")
        response = chat_agent.generate(query)
        print(response)