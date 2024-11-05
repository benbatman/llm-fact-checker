import requests

import os

from dotenv import load_dotenv, find_dotenv

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    ServiceContext,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding


load_dotenv(find_dotenv(), override=True)


# headers = {
#     "Accept": "application/json",
#     "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
#     "Content-Type": "application/json",
# }


# def query(payload):
#     response = requests.post(os.getenv("EMBEDDING_API"), headers=headers, json=payload)
#     return response.json()


# output = query(
#     {
#         "inputs": [
#             "This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music!",
#             "bing bong bong lasdfke d",
#             "cheese and rice",
#         ],
#         "parameters": {},
#     }
# )
# print(len(output))

persist_dir = "./vector_index"
embed_model = HuggingFaceEmbedding(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    query_instruction="Represent this sentence for searching relevant passages:",
    truncate_dim=512,
)


def load_vector_store(persist_dir: str) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
    )
    return load_index_from_storage(storage_context, embed_model=embed_model)


def retrieve(query: str, persist_dir: str) -> str:
    print("> RAG CALLED")
    index = load_vector_store(persist_dir)
    retriever = index.as_retriever()
    response = retriever.retrieve(query)
    return response


output = retrieve("mark wahlberg and disney", persist_dir)
print(output[0].text, output[0].metadata)
