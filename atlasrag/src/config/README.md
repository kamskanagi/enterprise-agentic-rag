# Configuration Module (Phase 2)

**Status:** Planning phase - to be implemented in Phase 2

**Purpose:** Centralized settings management and environment variable handling.

## Responsibilities

- Load environment variables from `.env` file using `python-dotenv`
- Validate configuration on application startup
- Provide type-safe configuration objects using Pydantic Settings
- Support multiple environments (dev, staging, production)
- Enable provider-specific settings (Ollama, OpenAI, Gemini)
- Manage secrets securely

## Design Pattern

All configuration is immutable once loaded (no runtime changes). Configuration is accessed via a singleton pattern:

```python
from src.config.settings import get_settings

settings = get_settings()  # Thread-safe singleton
```

## Key Files (to be implemented in Phase 2)

- `settings.py` - Main settings loader (Pydantic Settings)
- `llm_config.py` - LLM provider-specific configurations
- `vector_store_config.py` - Vector database configurations
- `api_config.py` - API server configurations
- `observability_config.py` - Logging and monitoring setup

## Configuration Options (from .env.example)

### LLM Provider
- `LLM_PROVIDER`: Which provider to use (ollama/openai/gemini)
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`
- `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`

### Vector Store
- `VECTOR_STORE`: Which vector DB (chroma/milvus)
- Chroma: `CHROMA_PERSIST_DIRECTORY`, `CHROMA_COLLECTION_NAME`
- Milvus: `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_USER`, `MILVUS_PASSWORD`

### Database
- `DATABASE_URL`: PostgreSQL connection string
- `DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW`

### API
- `API_HOST`, `API_PORT`, `API_WORKERS`, `API_RELOAD`
- `CORS_ALLOW_ORIGINS`, `CORS_ALLOW_CREDENTIALS`

### Agents & Retrieval
- `MAX_REPAIR_ITERATIONS`, `MIN_CITATION_COVERAGE`, `VERIFICATION_MODE`
- `RETRIEVAL_TOP_K`, `SIMILARITY_THRESHOLD`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`

### Observability
- `ENABLE_TRACING`, `ENABLE_METRICS`
- `LOG_LEVEL`, `LOG_FORMAT`
- `OTEL_EXPORTER_OTLP_ENDPOINT`

## Usage Example (Phase 2+)

```python
from src.config.settings import get_settings

# Get configuration
settings = get_settings()

# Access settings
llm_provider = settings.llm_provider  # "ollama" | "openai" | "gemini"
api_port = settings.api_port  # 8000
chunk_size = settings.chunk_size  # 1000

# Provider-specific settings
if settings.llm_provider == "openai":
    client = OpenAI(api_key=settings.openai_api_key)
elif settings.llm_provider == "ollama":
    client = Ollama(base_url=settings.ollama_base_url)
```

## Validation

Configuration is validated on load:
- Type checking: "api_port" must be int, not string
- Range checking: "similarity_threshold" must be 0.0-1.0
- Requirement checking: "database_url" is required in production
- Consistency checking: "max_repair_iterations" < "max_retries"

If validation fails, app startup fails with clear error messages.

## Environment-Specific Overrides

Configuration hierarchy (highest to lowest priority):
1. Environment variables (ATLASRAG_*)
2. Loaded from .env file
3. Default values in settings class
4. Pydantic Settings defaults

## Phase Dependencies

- **Phase 2**: Implement this module
- **Phase 3**: LLM module uses `settings.llm_provider`
- **Phase 4**: Retrieval module uses `settings.vector_store`
- **Phase 5**: Ingestion module uses `settings.chunk_size`
- **Phase 9**: API module uses `settings.api_port`, `settings.cors_*`
- **Phase 10**: Observability uses `settings.log_level`, `settings.enable_tracing`
- **Phase 11**: Evaluation uses `settings.eval_dataset_path`, `settings.ragas_metrics`

## Testing (Phase 2+)

```python
import pytest
from src.config.settings import Settings

def test_settings_loads_from_env(monkeypatch):
    """Test that settings loads environment variables"""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    settings = Settings()
    assert settings.llm_provider == "openai"

def test_settings_validates_chunk_size():
    """Test that chunk size must be positive"""
    with pytest.raises(ValueError):
        Settings(chunk_size=-1)
```

## Security Considerations

- Never log secrets (API keys) in debug output
- Use read-only .env file in docker-compose
- Validate secret presence in production env
- Support secret rotation via environment reload
- Use Pydantic's SecretStr for sensitive values

## Future Enhancements

- [ ] Multi-environment support (.env.dev, .env.prod)
- [ ] Secrets management integration (AWS Secrets Manager, etc.)
- [ ] Configuration hot-reload (without restart)
- [ ] Configuration schema documentation (JSON Schema)
- [ ] Configuration UI for admin panel (Phase 15)
