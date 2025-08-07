from agno.models.openai.like import OpenAILike

import mal.providers as mal


def model_by_provider_with_model(provider: mal.Provider, is_beta=False, model_name="") -> OpenAILike:
    base_url = provider.beta_base_url if is_beta else provider.base_url
    model_name = model_name if model_name else provider.model_id
    return OpenAILike(
        id=model_name,
        api_key=provider.api_key,
        base_url=base_url
    )


def model_by_provider(provider: mal.Provider, is_beta=False, model_type=mal.default_model_type) -> OpenAILike:
    model_name = provider.model_id_by_type(model_type)
    return model_by_provider_with_model(provider, is_beta, model_name)
