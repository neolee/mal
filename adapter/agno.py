from agno.models.openai.like import OpenAILike
from agno.embedder.ollama import OllamaEmbedder
from agno.embedder.openai import OpenAIEmbedder

from mal.providers import provider_by_alias
from mal.util import parse_model_str


## openai compatible model

def model(model: str) -> OpenAILike:
    provider_name, model_id = parse_model_str(model)
    provider = provider_by_alias(provider_name)
    model_id = model_id if model_id else provider.model_id

    return OpenAILike(
        id=model_id,
        api_key=provider.api_key,
        base_url=provider.base_url
    )


## openai compatible embedder

def openai_embedder(model: str, dimensions: int) -> OpenAIEmbedder:
    provider_name, model_id = parse_model_str(model)
    provider = provider_by_alias(provider_name)
    model_id = model_id if model_id else provider.model_id

    return OpenAIEmbedder(
        base_url=provider.base_url,
        api_key=provider.api_key,
        id=model_id,
        dimensions=dimensions
    )


## ollama embedder

def ollama_embedder(model_id: str, dimensions: int) -> OllamaEmbedder:
    return OllamaEmbedder(id=model_id, dimensions=dimensions)
