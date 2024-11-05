from nemoguardrails.actions import action
from prompts.fact_check import get_prompt
from llm_utils.perform_fact_check import perform_fact_check


@action(name="CustomCheckFactsAction")
async def check_facts(context: str, llm_answer: str, user_message: str) -> bool:
    """
    This function checks if the given context and LLM answer contain the same information.
    It returns True if they are the same, and False otherwise.
    """
    print("> CHECKING FACTS")
    search_results = llm_answer.split("SEARCH RESULTS:")[-1]
    prompt = get_prompt(context, llm_answer, search_results)
    response = perform_fact_check(prompt)

    return True if "YES" in response else False


# Alterative until updated to Colang 2.x
async def check_facts_v1(context: str, llm_answer: str, user_message: str) -> bool:
    """
    This function checks if the given context and LLM answer contain the same information.
    It returns True if they are the same, and False otherwise.
    """
    print("> CHECKING FACTS")
    print("LLM ANSWER: ", llm_answer)
    search_results = ""
    if "SEARCH RESULTS" in llm_answer:
        search_results = llm_answer.split("SEARCH RESULTS:")[-1]

    prompt = get_prompt(context, llm_answer, search_results, user_message)
    response = perform_fact_check(prompt)

    return True if "YES" in response else False
