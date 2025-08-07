from enum import Enum

import re
from textwrap import dedent

from openai import OpenAI, AsyncOpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from mal.providers import Provider, provider_by_alias
from mal.util import parse_model_str


## helper data type

class Difficulty(Enum):
    Easy = 1
    Hard = 2


## utilities

def _create_client(provider: Provider, is_beta: bool) -> OpenAI:
    base_url = provider.beta_base_url if is_beta else provider.base_url
    return OpenAI(base_url=base_url, api_key=provider.api_key)


## openai compatible client

class Client:
    def __init__(self, model: str, name="", is_beta=False) -> None:
        provider_name, model_id = parse_model_str(model)
        self.provider = provider_by_alias(provider_name)
        self.model = model_id if model_id else self.provider.model_id
        self.name = name
        self.is_beta = is_beta
        self.client = _create_client(self.provider, self.is_beta)

    def set_mode(self, is_beta=False):
        if self.is_beta == is_beta: return
        self.is_beta = is_beta
        self.client = _create_client(self.provider, self.is_beta)

    ## completions

    def create_chat_completion(self, messages: list, stream=False, auto_think=False, **kwargs) -> ChatCompletion | Stream[ChatCompletionChunk]:
        if auto_think:
            q = self.last_user_message(messages)
            if q:
                d = self.evaluate_difficulty(q)
                if d == Difficulty.Easy:
                    self.append_message(messages, "assistant", "<think>\n\n</think>\n\n")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        return response

    def create_completion(self, **kwargs):
        response = self.client.completions.create(model=self.model, **kwargs)
        return response

    ## response protocol helpers

    def append_message(self, messages: list, role: str, message: str):
        messages.append({"role": role, "content": message})

    def last_user_message(self, messages: list) -> str:
        for message in messages[::-1]:
            if message["role"] == "user": return message["content"]
        return ""

    def chat_completion_message(self, completion):
        return completion.choices[0].message

    def chat_completion_content(self, completion):
        return completion.choices[0].message.content

    def chat_completion_chunk_content(self, chunk):
        return chunk.choices[0].delta.content

    def chat_completion_reasoning_content(self, completion):
        try:
            return completion.choices[0].message.reasoning_content
        except:
            return ""

    def chat_completion_chunk_reasoning_content(self, chunk):
        try:
            return chunk.choices[0].delta.reasoning_content
        except:
            return ""

    def chat_completion_json(self, completion):
        return completion.choices[0].message.model_dump_json()

    def chat_completion_tool_calls(self, completion):
        return completion.choices[0].message.tool_calls

    def completion_text(self, completion):
        return completion.choices[0].text

    ## extra utilities

    def evaluate_difficulty(self, prompt: str) -> Difficulty:
        """Evaluate the difficulty of a user query.

        Args:
            prompt: User prompt string
            client: OpenAI compatible API client
            model_name: Model used for this evaluation
        Return:
            Difficulty.Easy if the model think it's easy, Difficulty.Hard otherwise
        """

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

        # first check for explicit thinking instructions
        if (forced_difficulty := detect_think_instruction(prompt)) is not None:
            return forced_difficulty

        # evaluate difficulty using given model
        system_message = "You are a specialized AI model acting as a request difficulty assessor."
        prompt_template = dedent("""\
            You are a specialized AI model acting as a request difficulty assessor.
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

            <users_request>\n{prompt}\n</users_request>
        """)
        prompt = prompt_template.format(prompt=prompt)

        completion = self.create_chat_completion(
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

        result = self.chat_completion_content(completion)
        match result:
            case "easy": return Difficulty.Easy
            case "hard": return Difficulty.Hard
            case _: return Difficulty.Hard


## openai compatible embedder

class Embedder:
    def __init__(self, model: str, dimensions: int) -> None:
        provider_name, model_id = parse_model_str(model)
        self.provider = provider_by_alias(provider_name)
        self.model = model_id
        self.dimensions = dimensions

        self.client = AsyncOpenAI(
            base_url=self.provider.base_url,
            api_key=self.provider.api_key
        )

    async def create_embedding(self, s: str):
        embedding = await self.client.embeddings.create(
            input=s,
            model=self.model,
            timeout=10.0
        )
        assert len(embedding.data) == 1, (
            f"expected 1 embedding, got {len(embedding.data)}, doc query: {s!r}"
        )
        return embedding.data[0].embedding
