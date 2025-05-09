from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

import mal.providers as mal
from mal.util.llm import Difficulty, evaluate_difficulty


## openai compatible client and model config

def client_by_provider(provider=mal.default_provider, is_beta=False) -> OpenAI:
    base_url = provider.beta_base_url if is_beta else provider.base_url
    return OpenAI(base_url=base_url, api_key=provider.api_key)


class Model:
    def __init__(self, provider: mal.Provider, model_name: str="", description="") -> None:
        self.provider = provider
        self.model_name = model_name if model_name else provider.model_id
        self.description = description


def model_by_provider_with_model(provider: mal.Provider, model_name="", description="") -> Model:
    if not model_name: model_name = provider.model_id
    return Model(provider, model_name, description)


def model_by_provider(provider: mal.Provider, model_type=mal.default_model_type, description="") -> Model:
    model_name = provider.model_id_from_type(model_type)
    return model_by_provider_with_model(provider, model_name, description)


deepseek = model_by_provider(mal.deepseek_provider, description="DeepSeek-V3")
deepseek_reasoner = model_by_provider(mal.deepseek_provider, "reasoner", "DeepSeek-R1")

qwen = model_by_provider(mal.qwen_provider, description="Qwen2.5-Max")
qwen_coder = model_by_provider(mal.qwen_provider, "coder", "Qwen2.5-Coder-Plus")
qwen_reasoner = model_by_provider(mal.qwen_provider, "reasoner", "QwQ-Plus")

openrouter = model_by_provider(mal.openrouter_provider, description="Gemini 2.5 Flash Preview")
openrouter_gemini_flash = model_by_provider_with_model(mal.openrouter_provider, "google/gemini-2.5-flash-preview", "Gemini 2.5 Flash Preview")
openrouter_gemini_pro = model_by_provider_with_model(mal.openrouter_provider, "google/gemini-2.5-pro-preview", "Gemini 2.5 Pro Preview")

local = model_by_provider(mal.local_provider, description="Gemma-3")
local_qwen = model_by_provider_with_model(mal.local_provider, "qwen3", "Qwen3-30B-A3B")
local_gemma = model_by_provider_with_model(mal.local_provider, "gemma-3", "Gemma-3-12B")

lmstudio = model_by_provider(mal.lmstudio_provider, description="LM Studio Default")

ollama = model_by_provider(mal.ollama_provider, description="Ollama Default")

default = qwen

models = [deepseek, deepseek_reasoner, qwen, qwen_coder, qwen_reasoner,
          openrouter_gemini_flash, openrouter_gemini_pro,
          local_qwen, local_gemma, lmstudio]


## openai compatible api helper functions

def append_message(messages: list, role: str, message: str):
    messages.append({"role": role, "content": message})


def last_user_message(messages: list) -> str:
    for message in messages[::-1]:
        if message["role"] == "user": return message["content"]
    return ""


def create_chat_completion(
        client: OpenAI,
        model_name: str,
        messages: list,
        stream=False,
        auto_think=False,
        **kwargs) -> ChatCompletion | Stream[ChatCompletionChunk]:
    if auto_think:
        q = last_user_message(messages)
        if q:
            d = evaluate_difficulty(q, client, model_name)
            if d == Difficulty.Easy:
                append_message(messages, "assistant", "<think>\n\n</think>\n\n")

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=stream,
        **kwargs
    )
    return response


def chat_completion_message(completion):
    return completion.choices[0].message


def chat_completion_content(completion):
    return completion.choices[0].message.content


def chat_completion_chunk_content(chunk):
    return chunk.choices[0].delta.content


def chat_completion_reasoning_content(completion):
    try:
        return completion.choices[0].message.reasoning_content
    except:
        return ""


def chat_completion_chunk_reasoning_content(chunk):
    try:
        return chunk.choices[0].delta.reasoning_content
    except:
        return ""


def chat_completion_json(completion):
    return completion.choices[0].message.model_dump_json()


def chat_completion_tool_calls(completion):
    return completion.choices[0].message.tool_calls


def create_completion(
        client: OpenAI,
        model_name: str,
        **kwargs):
    response = client.completions.create(
        model=model_name,
        **kwargs
    )
    return response


def completion_text(completion):
    return completion.choices[0].text
