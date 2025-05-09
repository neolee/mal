from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

import mal.providers as mal


def openai_provider(provider: mal.Provider, is_beta=False) -> OpenAIProvider:
    base_url = provider.beta_base_url if is_beta else provider.base_url
    return OpenAIProvider(base_url=base_url, api_key=provider.api_key)


def model_by_provider_with_model(provider: mal.Provider, is_beta=False, model_name="") -> OpenAIModel:
    op = openai_provider(provider, is_beta)
    if not model_name: model_name = provider.model_id
    return OpenAIModel(model_name, provider=op)


def model_by_provider(provider: mal.Provider, is_beta=False, model_type=mal.default_model_type) -> OpenAIModel:
    model_name = provider.model_id_from_type(model_type)
    return model_by_provider_with_model(provider, is_beta, model_name)


def ollama_model(model_name: str) -> OpenAIModel:
    op = openai_provider(mal.ollama_provider)
    return OpenAIModel(model_name, provider=op)


deepseek = model_by_provider(mal.ollama_provider)
deepseek_beta = model_by_provider(mal.ollama_provider, is_beta=True)
deepseek_reasoner = model_by_provider(mal.ollama_provider, model_type="reasoner")

qwen = model_by_provider(mal.qwen_provider)
qwen_coder = model_by_provider(mal.qwen_provider, model_type="coder")
qwen_reasoner = model_by_provider(mal.qwen_provider, model_type="reasoner")

openrouter = model_by_provider(mal.openrouter_provider)
openrouter_gemini_flash = model_by_provider_with_model(mal.openrouter_provider, model_name="google/gemini-2.5-flash-preview")
openrouter_gemini_pro = model_by_provider_with_model(mal.openrouter_provider, model_name="google/gemini-2.5-pro-preview")

local = model_by_provider(mal.local_provider)
local_qwen = model_by_provider_with_model(mal.local_provider, model_name="qwen3")
local_gemma = model_by_provider_with_model(mal.local_provider, model_name="gemma-3")

lmstudio = model_by_provider(mal.lmstudio_provider)

ollama = model_by_provider(mal.ollama_provider)

default = qwen
