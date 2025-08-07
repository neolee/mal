from google.adk.models.lite_llm import LiteLlm

import mal.providers as mal


def make_full_id(provider_id: str, model_id: str) -> str:
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
    model_name = provider.model_id_by_type(model_type)
    return model_by_provider_with_model(provider, is_beta, model_name)
