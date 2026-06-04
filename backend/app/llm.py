"""
Model-agnostic LLM and embeddings factory.

Switch LLM_PROVIDER in .env to swap models without touching application code,
an "isolated, swappable models" design that keeps the agent provider-neutral
and protects against vendor lock-in.

Supported LLM providers:
    anthropic       - Claude via the Anthropic API (structured output)
    openai          - GPT models via the OpenAI API (structured output)
    bedrock         - Claude via AWS Bedrock (structured output, isolated env)
    nvidia_deepseek - DeepSeek via the NVIDIA NIM API (free, hosted, text mode)
    nvidia_moonshot - Moonshot Kimi via the NVIDIA NIM API (free, hosted, text)
    ollama          - any local model via Ollama (free, offline, text mode)

Supported embeddings providers:
    huggingface - local sentence-transformers (free, offline)
    openai      - OpenAI text-embedding-3-small (hosted)
"""

import os

# Providers whose LangChain integration implements .with_structured_output().
# For everything else we prompt for JSON and parse it (see agent.py).
STRUCTURED_OUTPUT_PROVIDERS = {"anthropic", "openai", "bedrock"}


def current_provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").lower()


def supports_structured_output() -> bool:
    return current_provider() in STRUCTURED_OUTPUT_PROVIDERS


def get_llm(temperature: float = 0.0, api_key: str | None = None, model: str | None = None):
    """
    Return a chat model for the configured provider.

    api_key overrides the environment variable when supplied (bring-your-own-key).
    model overrides NVIDIA_MODEL when supplied (user-selectable model).
    """
    provider = current_provider()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
            temperature=temperature,
            max_tokens=3000,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=model or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            region_name=os.getenv("AWS_REGION", "eu-central-1"),
            model_kwargs={"temperature": temperature, "max_tokens": 3000},
        )

    if provider == "nvidia_deepseek":
        resolved_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No NVIDIA API key found. "
                "Pass your key in the X-Api-Key request header or set NVIDIA_API_KEY."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=resolved_key,
            base_url="https://integrate.api.nvidia.com/v1",
            model=model or os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
            temperature=temperature,
            max_tokens=3000,
        )

    if provider == "nvidia_moonshot":
        resolved_key = api_key or os.getenv("NVIDIA_API_KEY_MOONSHOT", os.getenv("NVIDIA_API_KEY"))
        if not resolved_key:
            raise ValueError(
                "No NVIDIA API key found. "
                "Pass your key in the X-Api-Key request header or set NVIDIA_API_KEY."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=resolved_key,
            base_url="https://integrate.api.nvidia.com/v1",
            model=model or os.getenv("NVIDIA_MODEL_MOONSHOT", "moonshotai/kimi-k2-instruct"),
            temperature=temperature,
            max_tokens=3000,
        )

    if provider == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=model or os.getenv("OLLAMA_MODEL", "mistral"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. Valid options: "
        "anthropic, openai, bedrock, nvidia_deepseek, nvidia_moonshot, ollama."
    )


def get_embeddings():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "huggingface").lower()

    if provider == "huggingface":
        # Runs fully locally, no API key, no cost.
        # Requires: pip install langchain-huggingface sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    raise ValueError(
        f"Unknown EMBEDDINGS_PROVIDER: '{provider}'. "
        "Valid options: huggingface, openai."
    )
