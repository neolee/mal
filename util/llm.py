# miscellaneous utilities related to large language model

from enum import Enum
import re
from openai import OpenAI

import mal.openai.client as c


# evaluate the difficulty of given user query, determine whether a reasoning (deep thinking) model
# is needed, or just a non-reasoning model for faster response speed and better cost efficiency
class Difficulty(Enum):
    Easy = 1
    Hard = 2


def detect_think_instruction(prompt: str) -> Difficulty | None:
    """Check for special thinking instructions in the query string.
    
    Args:
        prompt: User prompt string
    Returns:
        Difficulty.Hard if contains /think
        Difficulty.Easy if contains /no_think or /nothink
    """
    if re.search(r'(?<![^\s/])/think(?![^\s/])', prompt, re.IGNORECASE):
        return Difficulty.Hard
    if re.search(r'(?<![^\s/])/no_?think(?![^\s/])', prompt, re.IGNORECASE):
        return Difficulty.Easy
    return None


def evaluate_difficulty(prompt: str, client: OpenAI, model_id: str) -> Difficulty:
    """Evaluate the difficulty of a user query.

    Args:
        prompt: User prompt string
        client: OpenAI compatible API client
        model_name: Model used for this evaluation
    Return:
        Difficulty.Easy if the model think it's easy, Difficulty.Hard otherwise
    """

    # first check for explicit thinking instructions
    if (forced_difficulty := detect_think_instruction(prompt)) is not None:
        return forced_difficulty

    # evaluate difficulty using given model
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

<users_request>\n{prompt}\n</users_request>"""
    prompt = prompt_template.format(prompt=prompt)

    # If no special instructions found, proceed with AI evaluation
    completion = c.create_chat_completion(
        client, model_id,
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

    result = c.chat_completion_content(completion)
    match result:
        case "easy": return Difficulty.Easy
        case "hard": return Difficulty.Hard
        case _: return Difficulty.Hard


if __name__ == "__main__":
    from mal.providers import local_provider

    client = c.client_by_provider(local_provider)
    model_id = local_provider.model_id

    query_strings = [
        "Introduce yourself",
        "8.11和8.9哪个数字更大？",
        "Explain 'illiberal democracy'",
        "Write a Python program to solve 8-Queen problem"
    ]

    for q in query_strings:
        print(f"> evaluating difficulty of \"{q}\"")
        d = evaluate_difficulty(q, client, model_id)
        match d:
            case Difficulty.Hard:
                print("it's pretty hard, use a reasoning model")
            case Difficulty.Easy:
                print("it's easy, use a normal model for faster response")
