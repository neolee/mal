from agno.models.openai.like import OpenAILike
from agno.models.ollama.chat import Ollama
from agno.models.deepseek.deepseek import DeepSeek

from mal.providers import provider_by_alias


deepseek_provider = provider_by_alias("deepseek")
ollama_provider = provider_by_alias("ollama")
qwen_provider = provider_by_alias("qwen")
lmstudio_provider = provider_by_alias("lmstudio")

deepseek = DeepSeek(id=deepseek_provider.model_id)
deepseek_reasoner = DeepSeek(id=deepseek_provider.reasoner_model_id)

ollama = Ollama(ollama_provider.model_id)
ollama_coder = Ollama(ollama_provider.coder_model_id)

qwen = OpenAILike(
    id=qwen_provider.model_id,
    api_key=qwen_provider.api_key,
    base_url=qwen_provider.base_url
)
qwen_coder = OpenAILike(
    id=qwen_provider.coder_model_id,
    api_key=qwen_provider.api_key,
    base_url=qwen_provider.base_url
)
qwen_reasoner = OpenAILike(
    id=qwen_provider.reasoner_model_id,
    api_key=qwen_provider.api_key,
    base_url=qwen_provider.base_url
)

lmstudio = OpenAILike(
    id=lmstudio_provider.model_id,
    api_key=lmstudio_provider.api_key,
    base_url=lmstudio_provider.base_url
)

default = qwen
default_reasoner = deepseek_reasoner
