from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

import os


load_dotenv(find_dotenv(), override=True)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NGC_API_KEY"),
)


def perform_fact_check(prompt: str):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    completion = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=messages,
        temperature=0.1,
        top_p=0.7,
        max_tokens=10,
    )
    return completion.choices[0].message.content
