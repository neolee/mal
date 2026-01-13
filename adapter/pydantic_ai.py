from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from mal.providers import provider_by_alias
from mal.util import parse_model_str


def openai_provider(provider_name: str) -> OpenAIProvider:
    provider = provider_by_alias(provider_name)
    return OpenAIProvider(base_url=provider.base_url, api_key=provider.api_key)


def openai_model(model: str) -> OpenAIChatModel:
    provider_name, model_id = parse_model_str(model)
    provider = provider_by_alias(provider_name)
    model_id = model_id if model_id else provider.model_id

    op = openai_provider(provider_name)
    return OpenAIChatModel(model_id, provider=op)
