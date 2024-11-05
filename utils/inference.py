import os
import asyncio
import json

from dotenv import load_dotenv, find_dotenv
from nemoguardrails import LLMRails, RailsConfig
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore

# from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from openai import OpenAI

# import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from serpapi import GoogleSearch

from actions import check_facts_v1

from utils.PgClient import PgClient

load_dotenv(find_dotenv(), override=True)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NGC_API_KEY"),
)

# pg_client = PgClient(
#     user=os.getenv("PG_USERNAME"),
#     password=os.getenv("PG_PASSWORD"),
#     host=os.getenv("PG_HOST"),
#     db_name=os.getenv("PG_DB"),
# )

embed_model = HuggingFaceEmbedding(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    query_instruction="Represent this sentence for searching relevant passages:",
    truncate_dim=512,
    device="cpu",
)

dim = 512
persist_dir = "./vector_index"
faiss_persist_dir = "./faiss_vector_index"


# def load_faiss_vector_store(persist_dir: str) -> VectorStoreIndex:
#     vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
#     storage_context = StorageContext.from_defaults(
#         vector_store=vector_store, persist_dir=persist_dir
#     )
#     return load_index_from_storage(storage_context, embed_model=embed_model)


def load_vector_store(persist_dir: str) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
    )
    return load_index_from_storage(storage_context, embed_model=embed_model)


# Create index
print("> LOADING VECTOR STORE")
index = load_vector_store(persist_dir)


async def retrieve(query: str, top_k: int = 4) -> str:
    print("> RAG CALLED")
    retriever = index.as_retriever()
    response = retriever.retrieve(query)
    text = response[0].text
    metadata = response[0].metadata
    score = response[0].score
    return (text, metadata, score)


def generate_system_prompt(context: tuple) -> str:
    SYSTEM_PROMPT = f"""You are a fact checking bot. Your job is to take in the user query, question or statement and use the supplied context to the verify if the statement is true or not.

                    The metadata contains the url where the information can be found online, the date when the article came out, the title/claim content
                    and sometimes the claim rating (if the data is from Snopes.com) which rates the claim content. The claim rating could be something like true, false, fake, labeled satire, etc.
                    The metadata will not always have a claim rating. If it doesn't, use the context to determine if the statement is true or not.

                    You must use the context and metadata provided to fact check the user's question or statement. 

                    You have access to the following tool:

                    web_search(query: str) -> str)
                        This function performs a web search for the given query and returns relevant information

                    If the context/metadata does not contain a factual answer to the user's question, use the web_search tool to find additional information.
                    When providing the final answer, use the information obtained from the web search tool if you called it.
                    If the web search result information doesn't answer the question, just say "I don't know".

                    Context:
                    {context[0]}     

                    Metadata:
                    {context[1]}

                     If you are able to answer the question based off of the given context and metadata or after using the web_search tool, your response should be in the following JSON format:

                    {{
                        "answer": <your answer here>,
                        "reasoning": <your reasoning here>,
                        "sources": <urls>,
                        "date": <date>,
                    }}

                    """
    return SYSTEM_PROMPT


async def generate(prompt: str, context: dict) -> tuple:
    APPEND_PROMPT = "\n\nRemember, if you can't find the answer to the user's question in the context or metadata, use the web_search tool to find the latest information on the user's question."
    print("> GENERATE CALLED")
    print("CONTEXT: ", context)
    refined_context: tuple = context["context"]
    system_prompt = generate_system_prompt(refined_context)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Here is the user question/query: " + prompt + APPEND_PROMPT,
        },
    ]

    completion = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=messages,
        temperature=0.1,
        top_p=0.7,
        max_tokens=1024,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Performs a web search for the given query and returns relevant information about the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for on the web.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )

    print("> GENERATE COMPLETE")
    response_message = completion.choices[0].message
    message_dict = response_message.model_dump()

    if message_dict.get("tool_calls") is not None:
        function_name = message_dict.get("tool_calls")[0]["function"]["name"]
        arguments = json.loads(
            message_dict.get("tool_calls")[0]["function"]["arguments"]
        )

        if function_name == "web_search":
            search_query = arguments.get("query")
            search_results = await web_search(search_query)
            search_results = json.dumps(search_results)

            # Append the function result to the conversation
            messages.append(
                {"role": "function", "name": function_name, "content": search_results}
            )

            # Generate a new response including the function result
            second_completion = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=messages,
                temperature=0.1,
                top_p=0.7,
                max_tokens=1024,
            )

            return (
                second_completion.choices[0].message.content
                + "SEARCH RESULTS: "
                + search_results
            )
    else:
        return completion.choices[0].message.content


async def web_search(query: str) -> list[dict]:
    print("> WEB SEARCH CALLED")
    serp_api_key = os.environ.get("SERP_API_KEY")
    if not serp_api_key:
        raise ValueError("SERP_API_KEY environment variable is not set.")

    search = GoogleSearch(
        {
            "q": query,
            "api_key": serp_api_key,
            "num": 5,
        }
    )

    try:
        serp_results = search.get_dict()
    except Exception as e:
        print(f"Error: {e}")
        return "No web search results found."

    snippets = []
    for result in serp_results.get("organic_results", []):
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        source = result.get("source", "")
        if snippet:
            snippets.append({"snippet": snippet, "link": link, "source": source})

    if snippets:
        print("> SNIPPETS FOUND IN SEARCH RESULTS")
        return snippets

    else:
        print("> NO SNIPPETS FOUND IN SEARCH RESULTS")
        return "No web search results found."


print("> LOADING CONFIG")
config = RailsConfig.from_path("./config")

print("> INITIALIZING RAILS")
rails = LLMRails(config)

# Register the actions used in rag.co
print("> REGISTERING ACTIONS")
rails.register_action(action=retrieve, name="retrieve")
rails.register_action(action=generate, name="generate")
rails.register_action(action=web_search, name="web_search")
rails.register_action(action=check_facts_v1, name="check_facts")


async def main(prompt: str):
    output = await rails.generate_async(prompt=prompt)
    return output


if __name__ == "__main__":
    prompt = "Is halo season 3 coming out?"
    print("> PROMPT: ", prompt)
    print("> STARTING")
    response = asyncio.run(main(prompt))
    print("BOT RESPONSE: ", response)

    print("> EXPLAINING")
    info = rails.explain()
    print(info.print_llm_calls_summary())
    print(info.colang_history)
