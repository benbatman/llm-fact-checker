from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from datasets import load_dataset


from PgClient import PgClient
import torch
import requests
from dotenv import find_dotenv, load_dotenv

import os
import json
import time

load_dotenv(find_dotenv(), override=True)


pg_client = PgClient(
    user=os.getenv("PG_USERNAME"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    db_name=os.getenv("PG_DB"),
)


# is cuda available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")
truncate_dim = 512

mxbai_model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1", truncate_dim=truncate_dim
)

mxbai_prompt = "Represent this sentence for searching relevant passages: "

testing = True
if testing:
    query = "gaza strip"
    embedding = mxbai_model.encode(mxbai_prompt + query)
    print(embedding.shape)
    print("Number of bytes in float32: ", embedding.nbytes)

    # Binary Quantization
    # Note: must put brackets surrounding the embeddings when passing to the quantize_embeddings function
    binary_embeddings = quantize_embeddings([embedding], precision="ubinary")
    print(binary_embeddings[0])
    print("Number of bytes in ubinary: ", binary_embeddings[0].nbytes)
    print("Unpacked bits: ", np.unpackbits(binary_embeddings[0]))
    print("".join(map(str, np.unpackbits(binary_embeddings[0]))))
    print("Unpacked bits shape: ", np.unpackbits(binary_embeddings[0]).shape)

    ### Scalar (int8) quantization ###
    corpus = load_dataset("nq_open", split="train[:1000]")["question"]
    calibration_embeddings = mxbai_model.encode(corpus)
    embedding = mxbai_model.encode(query)
    int8_embeddings = quantize_embeddings(
        [embedding], precision="int8", calibration_embeddings=calibration_embeddings
    )
    print(int8_embeddings[0])
    print("Number of bytes in int8: ", int8_embeddings[0].nbytes)
    print(int8_embeddings[0].shape)

### HF Inference Endpoint ###
api = False
if api:
    API_URL = "https://hpr4jhppe9uyze4n.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json",
    }

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query(
        {
            "inputs": "This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music!",
            "parameters": {},
        }
    )

    print(output)

### Implement binary, int8 quantization into pg db ###
# Ensure pgvector is updated to 0.7 or higher
# pgvector has types halfvec and bit now

workflow = False
if workflow:
    # Workflow
    need_to_insert = True
    if need_to_insert:
        # 1. Get content data from pg
        pg_query = """SELECT cid, description FROM shows_meta"""
        rows = pg_client.query(pg_query)
        ids = [row[0] for row in rows]
        descriptions = [row[1] for row in rows]

        # 1.5 Create calibration embeddings for quantization (all descriptions should fit into the context window)
        calibration_embeddings = mxbai_model.encode(descriptions)

        # 2. Embed content into float32, binary, and scalar
        float32_embeddings = mxbai_model.encode(descriptions, normalize_embeddings=True)
        binary_embeddings = quantize_embeddings(
            float32_embeddings.reshape(1, -1), precision="ubinary"
        )
        unpacked_binary_embeddings = np.unpackbits(binary_embeddings[0])
        int8_embeddings = quantize_embeddings(
            float32_embeddings.reshape(1, -1),
            precision="int8",
            calibration_embeddings=calibration_embeddings,
        )

        # 4. Store embeddings in db (will store float32 for now but hopefully can remove later depending on performance)
        # Ensure columns mb_embeddings, mb_scalar_embeddings, and mb_binary_embeddings are created
        # Batch insert embeddings into db
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_float32 = float32_embeddings[i : i + batch_size]
            batch_binary = unpacked_binary_embeddings[i : i + batch_size]
            batch_scalar = int8_embeddings[i : i + batch_size]

            pg_client.insert_new_embeddings(
                table="shows_meta",
                data=list(zip(batch_ids, batch_float32, batch_binary, batch_scalar)),
            )

    # Start timer
    s_time = time.time()

    # 5. User sends in query, encode query to float32
    query = "NOVA"
    query_embedding = mxbai_model.encode(
        mxbai_prompt + query, normalize_embeddings=True
    )

    # 6. Quantize float32 to binary query
    binary_query = quantize_embeddings(
        query_embedding.reshape(1, -1), precision="ubinary"
    )
    unpacked_binary_query = np.unpackbits(binary_query[0])

    # 7. Search binary index, get content and unique ids using top_k and rescore_multiplier count
    top_k = 50
    rescore_multiplier = 10
    results = pg_client.search_binary_query(
        table="shows_meta",
        query_embedding=unpacked_binary_query,
        top_k=top_k * rescore_multiplier,
    )
    cids = [result[0] for result in results]

    # 8. Get corresponding scalar embeddings by id match
    id_query = (
        f"""SELECT cid, title, mb_scalar_embedding WHERE cid ANY ('{cids}::text[]')"""
    )
    scalar_rows = pg_client.query(id_query)
    titles = [row[1] for row in scalar_rows]
    mb_scalar_embeddings_results = [row[2] for row in scalar_rows]

    # 9. Rescore the top_k * rescore_multiplier using the float32 query embedding and the scalar document embeddings
    scores = query_embedding @ mb_scalar_embeddings_results.T
    print(scores)

    # 10. Sort the scores and return the top_k
    indices = np.argsort(scores)[::-1][:top_k]
    top_titles = [titles[i] for i in indices]
    top_k_scores = [scores[i] for i in indices]

    # End timer
    e_time = time.time()

    # Compare to standard non-quantized searching
