import os
import rtoml


with open("providers.toml", "r") as f:
    data = rtoml.load(f)

default_provider_name: str = data["defaults"]["provider"]
default_model_type: str = data["defaults"]["model_type"]


class Provider:
    def __init__(self, config: dict) -> None:
        self.description = config["description"]
        self.api_key = os.environ.get(config["api_key_name"], "")
        self.base_url = config["base_url"]
        self.beta_base_url = config["beta_base_url"] or self.base_url
        self.chat_model_id = config["chat_model_id"]
        self.coder_model_id = config["coder_model_id"]
        self.reasoner_model_id = config["reasoner_model_id"]

        model_type = default_model_type
        if "default_model_type" in config and config["default_model_type"]:
            model_type = config["default_model_type"]
        self.model_id = self.model_id_from_type(model_type)


    def model_id_from_type(self, model_type: str=default_model_type) -> str:
        match model_type:
            case "chat": return self.chat_model_id
            case "coder": return self.coder_model_id
            case "reasoner": return self.reasoner_model_id
            case _: return self.chat_model_id


_configs = data["providers"]
def provider_by_name(name: str=default_provider_name) -> Provider:
    config: dict = _configs[name]
    return Provider(config)

default_provider = provider_by_name()


_aliases = data["aliases"]
def provider_by_alias(alias: str) -> Provider:
    name = _aliases.get(alias, alias)
    return provider_by_name(name)


providers = []
for name in data["providers"]:
    providers.append(provider_by_name(name))


deepseek_provider = provider_by_alias("deepseek")
qwen_provider = provider_by_alias("qwen")
moonshot_provider = provider_by_alias("moonshot")
openrouter_provider = provider_by_alias("openrouter")
local_provider = provider_by_alias("local")
lmstudio_provider = provider_by_alias("lmstudio")
ollama_provider = provider_by_alias("ollama")
