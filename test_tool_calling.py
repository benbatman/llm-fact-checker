from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
from openai import OpenAI

import os
import json
import asyncio

from serpapi import GoogleSearch


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NGC_API_KEY"),
)


def generate_system_prompt() -> str:
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
                    "JD vance made fun of tim walz"   

                    Metadata:
                    {{
                        "url": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                        "date": "2020-01-01",
                        "title": "Avocado Cause: Mental Illness",
                        "claim_rating": "false"
                     }}

                    If you are able to answer the question based off of the given context and metadata or after using the web_search tool, your response should be in the following JSON format:

                    {{
                        "answer": <your answer here>,
                        "reasoning": <your reasoning here>,
                        "sources": <urls>,
                        "date": <date>,
                    }}

                    Your only output should be the JSON object, nothing else
                    """
    return SYSTEM_PROMPT


async def generate(prompt: str) -> str:
    APPEND_PROMPT = "\n\nRemember, if you can't find the answer to the user's question in the context or metadata, use the web_search tool to find the latest information on the user's question."
    print("> GENERATE CALLED")
    system_prompt = generate_system_prompt()
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

    response_message = completion.choices[0].message
    print("Printing response message: ", response_message)
    message_dict = response_message.model_dump()
    print("Printing message dict tool calls: ", message_dict.get("tool_calls"))

    if message_dict.get("tool_calls") is not None:
        function_name = message_dict.get("tool_calls")[0]["function"]["name"]
        arguments = json.loads(
            message_dict.get("tool_calls")[0]["function"]["arguments"]
        )

        if function_name == "web_search":
            search_query = arguments.get("query")
            search_results = await web_search(search_query)
            # Turn search results to string
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

            return second_completion.choices[0].message.content
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


output = asyncio.run(
    generate(
        "When is halo season 3 coming out?",
    )
)

print(output)
