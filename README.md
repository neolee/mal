# LLM Abstraction Layer (MAL)

*MAL (Language Model Abstraction Layer)* is a framework designed to streamline the configuration and integration of applications across various language model services. It includes a service provider configuration framework and adaptation layers tailored for popular agent frameworks.

Note: At present, *MAL* supports only OpenAI-compatible API service providers.

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
- `default_provider`
- `providers`
- `deepseek_provider` `qwen_provider` etc.

Most configuration options are self-explanatory; however, a few additional notes are worth mentioning:
- The group name without the `providers` prefix represents the provider's name (e.g., `deepseek`). You can use this name in the `provider_by_name` function to retrieve the corresponding `Provider` object.
- Using `provider_by_alias` instead of `provider_by_name` is always better since it falls back to `provider_by_name` when an `alias` does not exist.
- For security reasons, avoid storing real API keys directly in the configuration file. Instead, store these keys as environment variables and reference them by their names within the configuration file using the `api_key_name` field. The actual key will be accessible through the `Provider` objects when you retrieve them via the framework.
- Fields like `beta_base_url` and any model id other than `model_id` and `chat_model_id` are optional. Use these fields only if your provider supports the corresponding features.

## Adapter for OpenAI

> `mal/adapter/openai.py`

Simplified interfaces for interacting with the OpenAI RESTful API, designed primarily to achieve separation of concerns.

The `mal.adapter.openai.Client` class wraps OpenAI compatible client object and our features together. You need to assign a model string with `<provider name>/<model id>` style in the constructor. The second part of that string is optional and the default model id in `provider` will be used if it is empty.

At present, support is limited to the `OpenAI` client, while the development of the asynchronous `AsyncOpenAI` API is still ongoing.

An embedding model helper class `mal.adapter.openai.Embedder` is also included.

## Adapter for Pydantic AI

> `mal/adapter/pydantic_ai.py`

The [Pydantic AI framework](https://ai.pydantic.dev/) (`pydantic_ai`) constructs an `OpenAIProvider` object using the parameters `base_url` and `api_key`. This `OpenAIProvider` is then utilized along with `model_id` to create an `OpenAIModel` object, which is essential for constructing agents (via the `model=` parameter).

As a bridge between *MAL* and *Pydantic AI*, the module `mal.adapter.pydantic_ai` handles all the necessary tasks. Simply import this module and use `openai_model` function to create model objects which can be used in *Pydantic AI* framework. 

``` python
deepseek = openai_model("deepseek/deepseek-chat")
deepseek_reasoner = openai_model("deepseek/deepseek-reasoner")

qwen = openai_model("qwen/qwen-plus-latest")
qwen_coder = openai_model("qwen/qwen3-coder-plus")

openrouter_gemini_flash = openai_model("openrouter/google/gemini-2.5-flash")
openrouter_gemini_pro = openai_model("openrouter/google/gemini-2.5-pro")

default = qwen
```

``` python
from pydantic_ai import Agent
from models import default

hello_agent = Agent(
    model=default,
    system_prompt="Be concise, reply with one sentence.",
)
```

## Adapter for Agno

> `mal/adapter/agno.py`

Like the adapter for Pydantic AI, just use `model` function in `mal.adapter.agno` to create models which can be used in [Agno](https://github.com/agno-agi/agno) agent framework.

Another helper function `openai_embedder` also included for creating *Agno* compatible embedder models.
