import streamlit as st
import asyncio
import nest_asyncio
from datetime import date, datetime

nest_asyncio.apply()
import os
from dotenv import load_dotenv, find_dotenv

from nemoguardrails import LLMRails, RailsConfig
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore

# from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from openai import OpenAI
from serpapi import GoogleSearch

from actions import check_facts_v1


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
import re


load_dotenv(find_dotenv(), override=True)


@st.cache_resource
def init_constants():

    persist_dir = "./vector_index"

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get("NGC_API_KEY"),
    )

    embed_model = HuggingFaceEmbedding(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        query_instruction="Represent this sentence for searching relevant passages:",
        truncate_dim=512,
        device="cpu",
    )

    def load_vector_store(persist_dir: str) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
        )
        return load_index_from_storage(storage_context, embed_model=embed_model)

    # Create index, takes a hot minute
    print("> LOADING VECTOR STORE")
    index = load_vector_store(persist_dir)

    print("> LOADING CONFIG")
    config = RailsConfig.from_path("./config")

    print("> INITIALIZING RAILS")
    rails = LLMRails(config)

    return client, index, rails


client, index, rails = init_constants()


async def retrieve(query: str, top_k: int = 3) -> str:
    print("> RAG CALLED")
    retriever = index.as_retriever(top_k=top_k)
    response = retriever.retrieve(query)
    texts = []
    all_metadata = []
    for document in response:
        text = document.text
        metadata = document.metadata
        texts.append(text)
        all_metadata.append(metadata)
    return (texts[0], all_metadata[0], texts[1], all_metadata[1])


def format_context():
    pass


def generate_system_prompt(context: tuple) -> str:
    SYSTEM_PROMPT = f"""You are a fact checking bot. Your job is to take in the user query, question or statement and use the supplied context to the verify if the statement is true or not.

                    The metadata contains the url where the information can be found online, the date when the article came out, the title/claim content
                    and sometimes the claim rating (if the data is from Snopes.com) which rates the claim content. The claim rating could be something like true, false, fake, labeled satire, etc.
                    The metadata will not always have a claim rating. If it doesn't, use the context to determine if the statement is true or not.

                    You must use the context and metadata provided to fact check the user's question or statement. 

                    You have access to the following tool:

                    web_search(query: str) -> str)
                        This function performs a web search for the given query and returns relevant information

                    Here is the current date and time information:  
                    Todays date is: {date.today()}  
                    The current year is: {date.today().year}
                    The current month is: {date.today().month}
                    The current day is: {date.today().day}
                    The current hour is: {datetime.now().hour}

                    Source 1 context and metadata:
                    {context[0]}     
                    {context[1]}

                    Source 2 context and metadata: 
                    {context[2]}
                    {context[3]}
                        
                    If the context or metadata does not contain a factual answer to the user's question you must call the web_search tool to find additional information on the user's query.
                    When providing the final answer, use the information obtained from the web search tool if you called it.
                    If the web search result information doesn't answer the question, just say "I don't know".

                     If you are able to answer the question based off of the given context and metadata or after using the web_search tool, your response should be in the following valid JSON format.
                     Only include the valid JSON object in your response.

                    {{
                        "answer": <your answer here>,
                        "reasoning": <your reasoning for your answer here>,
                        "sources": <list of urls> (output should be a list of urls),
                        "date": <date>,
                    }}
                    """
    return SYSTEM_PROMPT


async def generate(prompt: str, context: dict) -> str:
    APPEND_PROMPT = """\n\nIf you can't fact check or verify the user's question/query with the given context or metadata, call the web_search tool to find the latest information on the user's question."""
    print("> GENERATE CALLED")
    # print("CONTEXT: ", context)
    refined_context: tuple = context["context"]
    # print("Refined context: ", refined_context)
    system_prompt = generate_system_prompt(refined_context)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{prompt} + {APPEND_PROMPT}",
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
        tool_choice={
            "type": "function",
            "function": {"name": "web_search"},
        },
    )

    response_message = completion.choices[0].message
    message_dict = response_message.model_dump()

    if message_dict.get("tool_calls", []):
        function_name = message_dict.get("tool_calls")[0]["function"]["name"]
        tool_id = message_dict.get("tool_calls")[0]["id"]
        arguments = json.loads(
            message_dict.get("tool_calls")[0]["function"]["arguments"]
        )

        if function_name == "web_search":
            search_query = arguments.get("query")
            search_results = await web_search(search_query)
            search_results = json.dumps(search_results)

            # Append the function result to the conversation
            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": search_results,
                    "tool_call_id": tool_id,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"""Use the search results now to answer the question if the search results contain enough information to answer their question.
                                Here is question you must try and answer: {prompt}. 
                                Here are the relevant search results you can use to answer the question: {search_results}. 
                                If you can answer the question with the information from the search results, give a definitive answer.
                                Be sure to check the dates of the search results to see if they are relevant to the question.
                                Be succinct in your answer but make sure to answer the question""",
                }
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
            "num": 6,
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
        source_date = result.get("date", "")
        if snippet:
            snippets.append(
                {
                    "snippet": snippet,
                    "link": link,
                    "source": source,
                    "date": source_date,
                }
            )

    if snippets:
        print("> SNIPPETS FOUND IN SEARCH RESULTS")
        # print("SNIPPETS: ", snippets)
        return snippets

    else:
        print("> NO SNIPPETS FOUND IN SEARCH RESULTS")
        return "No web search results found."


rails.register_action(action=retrieve, name="retrieve")
rails.register_action(action=generate, name="generate")
rails.register_action(action=check_facts_v1, name="check_facts_v1")


def get_response(prompt, messages=None):
    return asyncio.run(rails.generate_async(prompt, messages))


def update_message(message_placeholder, message: str):
    message_placeholder.text(message)


def main() -> None:
    st.title("Fact Checker")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    prompt = st.text_input("Enter your question or statement")

    if st.button("Submit"):
        if prompt:
            with st.spinner("Generating response..."):
                messages = st.session_state.messages
                response = get_response(prompt, messages)
                split_response = response.split("SEARCH RESULTS:")
                response = split_response[0].strip()
                search_results = []
                if len(split_response) > 1:
                    # Has search results
                    search_results = split_response[1].strip()
                    search_results = json.loads(search_results)
                    print("SEARCH RESULTS: ", search_results)

                append_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
                st.session_state.messages.extend(append_messages)

            # Handle if response is not json format
            if not response.startswith("{") and not response.endswith("}"):
                st.write(response)

            else:
                response = re.sub(r"^[^{]*({.*})[^}]*$", r"\1", response)
                response_dict = json.loads(response)
                answer = response_dict.get("answer", None)
                reasoning = response_dict.get("reasoning", None)
                sources = response_dict.get("sources", None)
                date = response_dict.get("date", None)

                st.write("**Answer:**")
                st.write(answer)
                st.write("**Reasoning:**")
                st.write(reasoning)
                st.write("**Sources:**")
                for link in sources:
                    print("LINK: ", link)
                    matched_result = next(
                        (result for result in search_results if result["link"] == link),
                        None,
                    )
                    if matched_result:
                        st.markdown(
                            f"[{link}]({link}) - {matched_result['source']}, {matched_result['date'] if matched_result['date'] else 'No date found in source'}",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"[{link}]({link})", unsafe_allow_html=True)

                if date and not search_results:
                    st.write("**Date of latest information received:**")
                    st.write(date)

        else:
            st.write("Please enter a prompt")


if __name__ == "__main__":
    main()
