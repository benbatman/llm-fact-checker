from PgClient import PgClient

import os
import json


from dotenv import load_dotenv, find_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers.quantization import quantize_embeddings
import numpy as np
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


load_dotenv(find_dotenv(), override=True)


pg_client = PgClient(
    user=os.getenv("PG_USERNAME"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    db_name=os.getenv("PG_DB"),
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {os.getenv("HF_TOKEN")}",
    "Content-Type": "application/json",
}


def query(payload):
    response = requests.post(os.getenv("EMBEDDING_API"), headers=headers, json=payload)
    return response.json()


print(f"Using device: {device}")
truncate_dim = 512

mxbai_model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1", truncate_dim=truncate_dim
)


create_tables = False
if create_tables:
    pg_client.create_tables()
    print("Tables created")


store_information = False
if store_information:
    store_pbs = True
    if store_pbs:
        with open("extra_pbs_results.json", "r") as f:
            pbs_json: list[dict] = json.load(f)
        print("Number of pbs items: ", len(pbs_json))

        # Remove null values from pbs_json
        pbs_json = [item for item in pbs_json if item]

        for item in tqdm(pbs_json):
            if item:
                if "transcript" in item.keys():
                    item["body_text"] = item.pop("transcript")
                if "post_date" in item.keys():
                    item["timestamp"] = item.pop("post_date")

        # Save to new pbs results file
        with open("extra_pbs_results_cleaned.json", "w") as f:
            json.dump(pbs_json, f, indent=4)

        # Get cleaned results
        with open("extra_pbs_results_cleaned.json", "r") as f:
            print("Loading new PBS results from file")
            pbs_json = json.load(f)

        corpus = []
        for item in pbs_json:
            if item:
                if item["body_text"] and item["title"]:
                    corpus.extend([item["title"], item["title"]])

        if os.path.exists("pbs_calibration_embeddings.npz"):
            print("Loading calibration embeddings from file")
            loaded_embeddings = np.load("utils/pbs_calibration_embeddings.npz")
            calibration_embeddings = loaded_embeddings["embeddings"]
        else:
            calibration_embeddings = mxbai_model.encode(corpus, show_progress_bar=True)
            np.savez(
                "utils/pbs_calibration_embeddings.npz",
                embeddings=calibration_embeddings,
            )

        print("Preparing to embed pbs data")
        data_to_insert = []
        BATCH_SIZE = 32
        for item in tqdm(pbs_json):
            if item:
                if item["body_text"] and item["title"]:
                    # Chunk the body text before embedding
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512, chunk_overlap=0
                    )
                    texts = text_splitter.split_text(item["body_text"])
                    texts.extend(item["title"])
                    print(len(texts))

                    # if len(texts) > BATCH_SIZE:
                    #     num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
                    #     print("Num batches: ", num_batches)
                    #     for i, batch in enumerate(range(0, len(texts), BATCH_SIZE)):
                    #         payload = {
                    #             "inputs": texts[batch : batch + BATCH_SIZE],
                    #             "parameters": {},
                    #         }
                    #         outputs = query(payload)
                    #         print(len(outputs))
                    #         if i == 0:
                    #             body_embedding = outputs

                    #         elif i < num_batches - 1:
                    #             body_embedding = np.concatenate(
                    #                 [body_embedding, outputs]
                    #             )
                    #         else:
                    #             print('Getting title')
                    #             body_embedding = np.concatenate([body_embedding, outputs[:-1]])
                    #             title_embedding = np.array(outputs[-1])

                    # else:
                    #     payload = {
                    #     "inputs": texts,
                    #     "parameters": {},
                    #     }
                    #     outputs = query(payload)
                    #     body_embedding = outputs[:-1]
                    #     title_embedding = outputs[-1]

                    body_embedding = mxbai_model.encode(
                        texts, normalize_embeddings=False
                    )
                    # # Combine each embedding by mean pooling
                    body_embedding = np.mean(body_embedding, axis=0, dtype=np.float64)
                    # # Normalize the embedding
                    body_embedding /= np.linalg.norm(body_embedding)
                    title_embedding = mxbai_model.encode(item["title"])

                    item["content_embedding"] = str(body_embedding.tolist())
                    item["title_embedding"] = str(title_embedding.tolist())

                    body_int8_embeddings = quantize_embeddings(
                        [body_embedding],
                        precision="int8",
                        calibration_embeddings=calibration_embeddings,
                    )[0].tolist()

                    item["scaled_content_embedding"] = str(body_int8_embeddings)

                    int8_embeddings = quantize_embeddings(
                        [title_embedding],
                        precision="int8",
                        calibration_embeddings=calibration_embeddings,
                    )[0].tolist()
                    item["scaled_title_embedding"] = str(int8_embeddings)

                    binary_title_embedding = np.unpackbits(
                        quantize_embeddings(
                            [title_embedding],
                            precision="ubinary",
                        )[0]
                    )
                    binary_title_embedding = "".join(map(str, binary_title_embedding))
                    item["binary_title_embedding"] = binary_title_embedding

                    binary_content_embedding = np.unpackbits(
                        quantize_embeddings(
                            [body_embedding],
                            precision="ubinary",
                        )[0]
                    )
                    binary_content_embedding = "".join(
                        map(str, binary_content_embedding)
                    )
                    item["binary_content_embedding"] = binary_content_embedding

                    data_to_insert.append(item)

        print("Finished embedding")
        pg_client.insert_data(table_name="pbs", data=data_to_insert)

    store_snopes = False
    if store_snopes:
        all_snopes_results = []
        with open("snopes_results_0.json", "r") as f:
            snopes_json_0 = json.load(f)

        with open("snopes_results_1.json", "r") as f:
            snopes_json_1 = json.load(f)

        with open("snopes_results_2.json", "r") as f:
            snopes_json_2 = json.load(f)

        all_snopes_results.extend(snopes_json_0)
        all_snopes_results.extend(snopes_json_1)
        all_snopes_results.extend(snopes_json_2)

        data_subset = all_snopes_results[:5]

        # Generate full text corpus, list of all claim_cont and article_text
        # TODO: Go through and rescape all non-traditional snopes articles
        corpus = []
        for item in all_snopes_results:
            if item:
                if item["claim_cont"] and item["article_text"] != "":
                    corpus.extend([item["claim_cont"], item["article_text"]])

        if os.path.exists("snopes_calibration_embeddings.npz"):
            print("Loading calibration embeddings from file")
            loaded_embeddings = np.load("snopes_calibration_embeddings.npz")
            calibration_embeddings = loaded_embeddings["embeddings"]
        else:
            calibration_embeddings = mxbai_model.encode(corpus, show_progress_bar=True)
            np.savez(
                "snopes_calibration_embeddings.npz", embeddings=calibration_embeddings
            )

        print("Preparing to embed data")
        data_to_insert = []
        for item in tqdm(all_snopes_results):
            if item:
                if item["claim_cont"] and item["article_text"] != "":
                    article_embedding = mxbai_model.encode(item["article_text"])
                    claim_cont_embedding = mxbai_model.encode(item["claim_cont"])

                    item["article_embedding"] = str(article_embedding.tolist())
                    item["claim_cont_embedding"] = str(claim_cont_embedding.tolist())

                    article_int8_embeddings = quantize_embeddings(
                        [article_embedding],
                        precision="int8",
                        calibration_embeddings=calibration_embeddings,
                    )[0].tolist()

                    item["scaled_article_embedding"] = str(article_int8_embeddings)

                    int8_embeddings = quantize_embeddings(
                        [claim_cont_embedding],
                        precision="int8",
                        calibration_embeddings=calibration_embeddings,
                    )[0].tolist()
                    item["scaled_claim_cont_embedding"] = str(int8_embeddings)
                    # Binary embeddings
                    # These must be turned into strings of 1s and 0s
                    binary_article_embedding = np.unpackbits(
                        quantize_embeddings([article_embedding], precision="ubinary")[0]
                    )
                    binary_article_embedding = "".join(
                        map(str, binary_article_embedding)
                    )
                    item["binary_article_embedding"] = binary_article_embedding
                    binary_claim_cont_embedding = np.unpackbits(
                        quantize_embeddings(
                            [claim_cont_embedding], precision="ubinary"
                        )[0]
                    )
                    binary_claim_cont_embedding = "".join(
                        map(str, binary_claim_cont_embedding)
                    )
                    item["binary_claim_cont_embedding"] = binary_claim_cont_embedding
                    data_to_insert.append(item)

        print("finished embedding")

        pg_client.insert_data(table_name="snopes", data=data_to_insert)


CREATE_VECTOR_STORE = True
pbs_documents = []
snopes_documents = []
if CREATE_VECTOR_STORE:
    # dim = 512
    # faiss_index = faiss.IndexFlatL2(dim)
    print("Creating vector store")
    pbs = True
    if pbs:
        sql_query = "SELECT * FROM pbs"
        pbs_data = pg_client.query(sql_query)

        pbs_documents = [
            Document(
                text=row[4],
                embedding=row[5],
                metadata={
                    "title": row[1],
                    "url": row[3],
                    "timestamp": str(row[2]),
                    "source": "pbs",
                },
            )
            for row in pbs_data
        ]

    snopes = True
    if snopes:
        snopes_query = "SELECT * FROM snopes"
        snopes_data = pg_client.query(snopes_query)
        print(len(snopes_data))
        snopes_documents = [
            Document(
                text=row[5],
                embedding=row[6],
                metadata={
                    "title": row[1],
                    "url": row[4],
                    "timestamp": str(row[3]),
                    "claim_rating": row[2],
                    "source": "snopes",
                },
            )
            for row in snopes_data
        ]

    # Combine documents
    documents = pbs_documents + snopes_documents

    embed_model = HuggingFaceEmbedding(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        query_instruction="Represent this sentence for searching relevant passages: ",
        truncate_dim=512,
    )

    # print("Using FAISS index")
    # vector_store = FaissVectorStore(faiss_index=faiss_index)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    # index.storage_context.persist(
    #     persist_dir="./faiss_vector_index",
    # )

    print("Using standard index")
    pbs_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    pbs_index.storage_context.persist(
        persist_dir="./vector_index_v2",
    )
