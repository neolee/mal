# LLM Abstraction Layer (MAL)

MAL (Language Model Abstraction Layer) is a framework designed to streamline the configuration and integration of applications across various language model services. It includes a service provider configuration framework and adaptation layers tailored for popular agent frameworks.

Note: At present, MAL supports only OpenAI-compatible API service providers.

## Installation

Just add this repository as a submodule to your project. *In your project root directory* run the following command:

```shell
git submodule add https://github.com/neolee/mal.git
```

Next, copy the `mal/providers.toml` file to your project's root directory and customize the configurations as necessary. Alternatively, create a symbolic link in your project's root directory if you prefer to use the predefined configuration.

Lastly, include any required dependencies for your project. MAL itself needs `rtoml` for `.toml` configuration parsing. Depending on your adapter requirements, also add `openai`, `pydantic-ai`, or `agno`.

## Service Provider Configuration

> `providers.toml` `mal/providers.py`

- `providers.toml`: Contains multiple provider configurations that you can access by name or alias, along with other extensible attributes.
- `providers.py`: A Python wrapper for the configuration file, offering easy access to provider settings and additional options.

To use this framework, simply add your OpenAI-compatible service provider as a configuration group as shown below:

``` toml
[providers.deepseek]
description = "DeepSeek Official"
api_key_name = "DEEPSEEK_API_KEY"
base_url = "https://api.deepseek.com"
beta_base_url = "https://api.deepseek.com/beta"
chat_model_id = "deepseek-chat"
coder_model_id = "deepseek-chat"
reasoner_model_id = "deepseek-reasoner"
```

You can define aliases within the `[aliases]` section or modify system default
settings in the `[defaults]` section. Afterward, you can import `mal.providers`
and utilize the following functions and variables for managing `Provider`
objects:

- `provider_by_name`
- `provider_by_alias`
- `providers`
- `default_provider`

Most configuration options are self-explanatory; however, a few additional notes are worth mentioning:
- The group name without the `providers` prefix represents the provider's name (e.g., `deepseek`). You can use this name in the `provider_by_name` function to retrieve the corresponding `Provider` object.
- For security reasons, avoid storing real API keys directly in the configuration file. Instead, store these keys as environment variables and reference them by their names within the configuration file using the `api_key_name` field. The actual key will be accessible through the `Provider` objects when you retrieve them via the framework.
- Fields like `beta_base_url` and any model id other than `model_id` and `chat_model_id` are optional. Use these fields only if your provider supports the corresponding features.

## Model Adapter for OpenAI

> `mal/openai/model.py` `mal/openai/embedder.py`

Simplified interfaces for interacting with the OpenAI RESTful API, designed primarily to achieve separation of concerns.

Use `client_by_provider` to get OpenAI compatible `client` object from a `provider`. Use `model_by_provider_with_model` and `model_by_provider` to get a `Model` object including `provider`, `model_name` and `description` attributes. Or use predefined `Model` objects in `mal/openai/model.py`.

At present, support is limited to the `OpenAI` client, while the development of the asynchronous `AsyncOpenAI` API is still ongoing.

An embedding model helper `mal.openai.embedder.Embedder` is also included.

## Model Adapter for PydanticAI

> `mal/pydantic_ai/model.py`

The [PydanticAI framework](https://ai.pydantic.dev/) (`pydantic_ai`) constructs an `OpenAIProvider` object using the parameters `base_url` and `api_key`. This `OpenAIProvider` is then utilized along with `model_name` to create an `OpenAIModel` object, which is essential for constructing agents (via the `model=` parameter).

As a bridge between **MAL** and `pydantic_ai`, the module `mal.pydantic_ai.model` handles all the necessary tasks. Simply import this module and use functions like below:

- `model_by_provider_with_model`
- `model_by_provider`
- `ollama_model`

...or any pre-defined model objects such as:

``` python
deepseek = model_by_provider(mal.ollama_provider)
deepseek_beta = model_by_provider(mal.ollama_provider, is_beta=True)
deepseek_reasoner = model_by_provider(mal.ollama_provider, model_type="reasoner")

qwen = model_by_provider(mal.qwen_provider)
qwen_coder = model_by_provider(mal.qwen_provider, model_type="coder")
qwen_reasoner = model_by_provider(mal.qwen_provider, model_type="reasoner")

openrouter_gemini_flash = model_by_provider_with_model(mal.openrouter_provider, model_name="google/gemini-2.5-flash-preview")
openrouter_gemini_pro = model_by_provider_with_model(mal.openrouter_provider, model_name="google/gemini-2.5-pro-preview")

local_qwen = model_by_provider_with_model(mal.local_provider, model_name="qwen3")
local_gemma = model_by_provider_with_model(mal.local_provider, model_name="gemma-3")

default = qwen
```

``` python
from pydantic_ai import Agent
import mal.pydantic_ai.model as model

hello_agent = Agent(
    model=model.default,
    system_prompt="Be concise, reply with one sentence.",
)
```

## Model Adapter for Agno

> `mal/agno/model.py`

Use models defined in the above file for [agno](https://github.com/agno-agi/agno) agent framework.

## Model Adapter for Google ADK

> `mal/adk/model.py`

Use models defined in the above file for Google's [adk-python](https://github.com/google/adk-python). Additionally, the helper function `model_by_provider` can be used to create a `model` object from a MAL `provider`.
