from google.adk.models.lite_llm import LiteLlm

import mal.providers as mal


def _make_id(provider_id: str, model_id: str) -> str:
    return provider_id + "/" + model_id


def model_by_provider_with_model(provider: mal.Provider, is_beta=False, model_name="") -> LiteLlm:
    base_url = provider.beta_base_url if is_beta else provider.base_url
    model_name = model_name if model_name else provider.model_id
    return LiteLlm(
        base_url=base_url,
        api_key=provider.api_key,
        model=model_name
    )


def model_by_provider(provider: mal.Provider, is_beta=False, model_type=mal.default_model_type) -> LiteLlm:
    model_name = provider.model_id_from_type(model_type)
    return model_by_provider_with_model(provider, is_beta, model_name)


deepseek = LiteLlm(model=_make_id("deepseek", mal.deepseek_provider.model_id))
deepseek_reasoner = LiteLlm(model=_make_id("deepseek", mal.deepseek_provider.reasoner_model_id))

qwen = model_by_provider(mal.qwen_provider)
qwen_coder = model_by_provider(mal.qwen_provider, model_type="coder")
qwen_reasoner = model_by_provider(mal.qwen_provider, model_type="reasoner")

openrouter = model_by_provider(mal.openrouter_provider)
openrouter_gemini_flash = model_by_provider_with_model(mal.openrouter_provider, model_name="google/gemini-2.5-flash-preview")
openrouter_gemini_pro = model_by_provider_with_model(mal.openrouter_provider, model_name="google/gemini-2.5-pro-preview")

local = model_by_provider(mal.local_provider)

lmstudio = model_by_provider(mal.lmstudio_provider)

ollama = model_by_provider(mal.ollama_provider)

default = qwen
default_reasoner = deepseek_reasoner
