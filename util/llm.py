# miscellaneous utilities related to large language model

from enum import Enum
from openai import OpenAI
import mal.openai.model as openai


# evaluate the difficulty of given user query, determine whether a reasoning (deep thinking) model
# is needed, or just a non-reasoning model for faster response speed and better cost efficiency
class Difficulty(Enum):
    Easy = 1
    Hard = 2


def evaluate_difficulty(q: str, client: OpenAI, model_name: str) -> Difficulty:
    """evaluate the difficulty of a user query, return `easy` or `hard`

    args:
    q: the user query string
    client: OpenAI compatible API client
    model_name: specific model used for this evaluation
    """

    system_message = "You are a specialized AI model acting as a request difficulty assessor."
    prompt_template = """You are a specialized AI model acting as a request difficulty assessor.
Your SOLE and ONLY task is to evaluate the inherent difficulty of a user's request that is intended for another AI.
You will receive a user's request message.
Your objective is to determine if this request requires careful, deliberate thought from the downstream AI, or if it's straightforward.

Criteria for your decision:
1. If the user's request is complex, nuanced, requires multi-step reasoning, creative generation, in-depth analysis, or careful consideration by the AI to produce a high-quality response, you MUST respond with: `hard`
2. If the user's request is simple, factual, straightforward, or can likely be answered quickly and directly by the AI with minimal processing or deliberation, you MUST respond with: `easy`

IMPORTANT:
- Your response MUST be EXACTLY one of the two commands: `hard` and `easy`
- Do NOT add any other text, explanations, or pleasantries.
- Your assessment is about the processing difficulty for the *AI that will ultimately handle the user's request*.

---

### User's request:

<users_request>\n{query}\n</users_request>"""
    prompt = prompt_template.format(query=q)

    completion = openai.create_chat_completion(
        client, model_name,
        [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": "<think>\n\n</think>\n\n"
            }
        ]
    )

    result = openai.chat_completion_content(completion)
    match result:
        case "easy": return Difficulty.Easy
        case "hard": return Difficulty.Hard
        case _: return Difficulty.Hard


if __name__ == "__main__":
    from mal.providers import provider_by_alias

    p = provider_by_alias("local")
    client = openai.client_by_provider(p)
    model_name = p.model_id

    query_strings = [
        "Introduce yourself",
        "8.11和8.9哪个数字更大？",
        "Explain 'illiberal democracy'",
        "Write a Python program to solve 8-Queen problem"
    ]

    for q in query_strings:
        print(f"> evaluating difficulty of \"{q}\"")
        d = evaluate_difficulty(q, client, model_name)
        match d:
            case Difficulty.Hard:
                print("it's pretty hard, use a reasoning model")
            case Difficulty.Easy:
                print("it's easy, use a normal model for faster response")
