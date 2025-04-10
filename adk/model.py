from google.adk.models.lite_llm import LiteLlm

from mal.providers import Provider, default_model_type, provider_by_alias


deepseek_provider = provider_by_alias("deepseek")
ollama_provider = provider_by_alias("ollama")
qwen_provider = provider_by_alias("qwen")
lmstudio_provider = provider_by_alias("lmstudio")


def _make_id(provider_id: str, model_id: str) -> str:
    return provider_id + "/" + model_id


def model_by_provider(provider: Provider, is_beta=False, model_type=default_model_type) -> LiteLlm:
    base_url = provider.beta_base_url if is_beta else provider.base_url
    model_id = provider.model_id_from_type(model_type)
    return LiteLlm(
        base_url=base_url,
        api_key=provider.api_key,
        model=model_id
    )


deepseek = LiteLlm(model=_make_id("deepseek", deepseek_provider.model_id))
deepseek_reasoner = LiteLlm(model=_make_id("deepseek", deepseek_provider.reasoner_model_id))

ollama = LiteLlm(model=_make_id("ollama", ollama_provider.model_id))
ollama_coder = LiteLlm(model=_make_id("ollama", ollama_provider.coder_model_id))

qwen = model_by_provider(qwen_provider)
qwen_coder = model_by_provider(qwen_provider, model_type="coder")
qwen_reasoner = model_by_provider(qwen_provider, model_type="reasoner")

lmstudio = model_by_provider(lmstudio_provider)

default = qwen
default_reasoner = deepseek_reasoner
