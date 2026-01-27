from typing import List


class OpenRouterModelRegistry:
    """Central registry for OpenRouter model ids used by this benchmark."""

    MODELS = [
        "bytedance-seed/seed-1.6-flash",
        "google/gemini-3-flash-preview",
        "openai/gpt-5-mini",
        "openai/o3-mini",
        "meta-llama/llama-4-maverick",
        "anthropic/claude-3-haiku",
        "x-ai/grok-4.1-fast",
        "deepseek/deepseek-v3.2",
        "nvidia/nemotron-3-nano-30b-a3b",
        "mistralai/ministral-14b-2512",
    ]

    @classmethod
    def all(cls) -> List[str]:
        return list(cls.MODELS)
