def get_prompt(
    context: str, llm_answer: str, search_results: str, user_message: str
) -> str:
    SYSTEM_PROMPT = f"""
                    Your task is to determine if the LLMs answer aligns with the context information and/or search results given.
                    If the answer is correct and factually aligns with your final answer, return "YES". Otherwise, return "NO".
                    Do not return anything else.

                    Here is the user's question:
                    {user_message}

                    Here is the context:
                    {context}

                    Here are the web search results (may be empty if search tool wasn't called):
                    {search_results}

                    Here is the LLM's answer:
                    {llm_answer}

                    Again, If the LLM's answer or reasoning is correct and factually aligns with the given context/search results, return "YES". Otherwise, return "NO".
                    """
    return SYSTEM_PROMPT
