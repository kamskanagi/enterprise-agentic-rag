"""
LLM Abstraction Layer

TODO: Phase 3 - Unified interface for multiple LLM providers

Supported Providers:
- Ollama: Local, privacy-preserving LLM runtime
- OpenAI: Cloud-based GPT models
- Gemini: Google's generative AI models

This layer provides a consistent interface for all providers:
- generate(prompt: str) -> str: Generate text responses
- embed(text: str) -> List[float]: Create embeddings for semantic search

Key files to be implemented:
- base.py: Abstract base class for LLM providers
- ollama_provider.py: Ollama implementation
- openai_provider.py: OpenAI implementation
- gemini_provider.py: Gemini implementation
- factory.py: Provider selection factory

Usage example (Phase 3+):
    from src.llm.factory import get_llm_client
    llm = get_llm_client()  # Picks based on config
    response = llm.generate("What is 2+2?")
    embeddings = llm.embed("Hello world")

Design Pattern:
All providers implement the same interface defined in base.py,
allowing seamless switching based on configuration.
"""
