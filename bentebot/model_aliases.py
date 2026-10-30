"""Explicit deployment aliases; spelling alone never determines a provider."""

GPT_COMPAT_ALIASES = {
    "claude-astra-6": "gpt-6-astra",
    "claude-luna-6": "gpt-6-luna",
    "claude-sol-6.1": "gpt-6.1-sol",
    "claude-terra-6.1": "gpt-6.1-terra",
}
OPENAI_MODELS = tuple(GPT_COMPAT_ALIASES.values())
CLAUDE_MODELS = (
    "claude-haiku-4-5",
    "claude-haiku-5-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
)


def canonical_model(model: str) -> str:
    return GPT_COMPAT_ALIASES.get(model, model)
