# LLM Abstraction Layer (Phase 3)

**Status:** Planning phase - to be implemented in Phase 3

**Purpose:** Unified interface for multiple LLM providers (Ollama, OpenAI, Gemini).

## Supported Providers

| Provider | Type | Cost | Speed | Quality | Privacy |
|----------|------|------|-------|---------|---------|
| **Ollama** | Local | Free | Fast | Good | ✅ Offline |
| **OpenAI** | Cloud | $$$ | Medium | Excellent | ❌ Uploaded |
| **Gemini** | Cloud | $ | Fast | Very Good | ❌ Uploaded |

## Design Pattern

All providers implement the same interface defined in `base.py`:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Create an embedding for semantic search"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is accessible"""
        pass
```

## Factory Pattern

Provider selection is handled by a factory:

```python
from src.llm.factory import get_llm_client

llm = get_llm_client()  # Picks based on settings.llm_provider
response = llm.generate("What is 2+2?")
embeddings = llm.embed("Hello world")
```

## Key Files (to be implemented in Phase 3)

- `base.py` - Abstract base class defining the LLM interface
- `ollama_provider.py` - Ollama implementation (local LLM)
- `openai_provider.py` - OpenAI implementation (GPT-4, etc.)
- `gemini_provider.py` - Google Gemini implementation
- `factory.py` - Provider selection factory
- `exceptions.py` - Custom exceptions (ProviderError, etc.)
- `models.py` - Type definitions (LLMResponse, etc.)

## Usage Examples (Phase 3+)

### Basic Text Generation
```python
from src.llm.factory import get_llm_client

llm = get_llm_client()

# Simple prompt
response = llm.generate("Capital of France?")
print(response)  # "Paris"

# With parameters
response = llm.generate(
    prompt="Explain quantum computing",
    max_tokens=500,
    temperature=0.7
)
```

### Embeddings for Semantic Search
```python
llm = get_llm_client()

# Generate embeddings for documents
doc_text = "The quick brown fox jumps over the lazy dog"
embedding = llm.embed(doc_text)
print(len(embedding))  # ~1536 dimensions for OpenAI, varies for others

# Use embeddings for similarity search
query = "Quick fox"
query_embedding = llm.embed(query)
# Compare query_embedding to doc_embedding with cosine similarity
```

### Error Handling
```python
from src.llm.exceptions import ProviderUnavailable, LLMError

try:
    response = llm.generate("What is AI?")
except ProviderUnavailable:
    print("LLM provider is offline")
except LLMError as e:
    print(f"LLM error: {e}")
```

## Configuration Dependencies

Relies on Phase 2 configuration:
- `settings.llm_provider` - Which provider to use
- `settings.ollama_base_url`, `settings.ollama_model` - Ollama config
- `settings.openai_api_key`, `settings.openai_model` - OpenAI config
- `settings.gemini_api_key`, `settings.gemini_model` - Gemini config

## Provider Specifics

### Ollama (Local)
- **URL:** http://localhost:11434
- **Models:** llama2, mistral, neural-chat, dolphin-mixtral
- **Embedding Model:** nomic-embed-text
- **Speed:** Fast (no network)
- **Cost:** Free
- **Privacy:** Offline, data stays local

### OpenAI (Cloud)
- **Endpoint:** https://api.openai.com/v1
- **Models:** gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo
- **Embedding Model:** text-embedding-3-small, text-embedding-3-large
- **Speed:** Medium (network dependent)
- **Cost:** Per-token billing ($$)
- **Privacy:** Data sent to OpenAI servers

### Google Gemini (Cloud)
- **Endpoint:** https://generativelanguage.googleapis.com
- **Models:** gemini-1.5-pro, gemini-1.5-flash
- **Embedding Model:** models/embedding-001
- **Speed:** Fast
- **Cost:** Per-request billing ($)
- **Privacy:** Data sent to Google servers

## Retries & Error Handling

All providers should implement retry logic:
- Exponential backoff for rate limits
- Max 3 retries by default
- Timeout handling (default 30s)
- Clear error messages

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def _call_ollama(prompt: str) -> str:
    # Implementation
    pass
```

## Testing (Phase 3+)

```python
import pytest
from src.llm.factory import get_llm_client
from src.llm.ollama_provider import OllamaProvider

@pytest.fixture
def llm_client():
    return get_llm_client()

def test_generate_text(llm_client):
    """Test text generation"""
    response = llm_client.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0

def test_embed_text(llm_client):
    """Test embedding generation"""
    embedding = llm_client.embed("Hello world")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

def test_provider_unavailable():
    """Test graceful handling when provider is offline"""
    provider = OllamaProvider(base_url="http://invalid:99999")
    assert not provider.is_available()
```

## Integration with Other Modules

- **Phase 4 (Retrieval):** Uses `llm.embed()` to create document embeddings
- **Phase 5 (Ingestion):** Uses `llm.embed()` during chunking
- **Phase 7 (Agents):** Uses `llm.generate()` for agent responses
- **Phase 9 (API):** Exposes `llm.generate()` via `/query` endpoint
- **Phase 11 (Evaluation):** Uses `llm.generate()` for evaluation

## Future Enhancements

- [ ] Streaming responses for long outputs
- [ ] Batch processing for multiple prompts
- [ ] Token counting before sending (to avoid overages)
- [ ] Function calling/tool use support
- [ ] Custom model fine-tuning support
- [ ] Cost tracking and budgeting
- [ ] Provider health monitoring
