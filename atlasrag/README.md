# AtlasRAG Source Code

Welcome to the AtlasRAG implementation! This directory contains all the code for the Enterprise Agentic RAG Platform.

## Directory Structure

```
atlasrag/
├── src/              # Application source code
│   ├── config/       # Configuration and settings management (Phase 2)
│   ├── llm/          # LLM provider abstraction layer (Phase 3)
│   ├── retrieval/    # Vector database operations (Phase 4)
│   ├── ingestion/    # Document processing pipeline (Phase 5)
│   ├── agents/       # LangGraph agent implementations (Phase 7)
│   ├── api/          # FastAPI endpoints and routes (Phase 9)
│   └── evaluation/   # Quality metrics and RAGAS integration (Phase 11)
├── infra/            # Infrastructure as code
│   ├── docker/       # Docker and Docker Compose configurations
│   └── k8s/          # Kubernetes manifests (Phase 14)
├── eval/             # Evaluation datasets and golden Q&A pairs
├── scripts/          # Utility and helper scripts
└── tests/            # Test suites
    ├── unit/         # Unit tests
    └── integration/  # Integration tests
```

## Development Phases

This codebase is built incrementally following the 15-phase roadmap from [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md):

### Completed ✅
- **Phase 0**: Prerequisites and setup
- **Phase 1**: Project skeleton (current - you are here)

### Planned 🔄
- **Phase 2**: Configuration system
- **Phase 3**: LLM abstraction layer
- **Phase 4**: Vector store layer
- **Phase 5**: Document ingestion pipeline
- **Phase 6**: Basic RAG (retrieval + response)
- **Phase 7**: Agent architecture with LangGraph
- **Phase 8**: Verification & repair loops
- **Phase 9**: FastAPI server
- **Phase 10**: Observability (OpenTelemetry + Prometheus)
- **Phase 11**: Evaluation suite (RAGAS)
- **Phase 12**: CI/CD quality gates
- **Phase 13**: Docker & local demo
- **Phase 14**: Kubernetes manifests
- **Phase 15**: Polish & documentation

## Key Modules

### Configuration (Phase 2)
Centralized settings management with environment variable loading and type-safe settings objects.
See [src/config/README.md](src/config/README.md)

### LLM Abstraction Layer (Phase 3)
Unified interface for multiple LLM providers (Ollama, OpenAI, Gemini).
See [src/llm/README.md](src/llm/README.md)

### Retrieval (Phase 4)
Vector database operations for semantic search (Chroma, Milvus).
See [src/retrieval/README.md](src/retrieval/README.md)

### Ingestion (Phase 5)
Document processing pipeline (load, clean, chunk, embed, store).
See [src/ingestion/README.md](src/ingestion/README.md)

### Agents (Phase 7)
Multi-agent system with LangGraph (Planner, Retriever, Answerer, Verifier).
See [src/agents/README.md](src/agents/README.md)

### API (Phase 9)
FastAPI HTTP endpoints for querying and document ingestion.
See [src/api/README.md](src/api/README.md)

### Evaluation (Phase 11)
RAGAS-based quality metrics and evaluation framework.
See [src/evaluation/README.md](src/evaluation/README.md)

## Getting Started

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Install Development Tools
```bash
pip install -e ".[dev]"
```

### 3. Start Local Services
```bash
docker compose up
```

### 4. Run Tests
```bash
pytest atlasrag/tests/
```

### 5. Format & Lint
```bash
black atlasrag/
ruff check atlasrag/
mypy atlasrag/
```

## File Organization Notes

- Each module contains TODO comments indicating which phase implements it
- Use relative imports within the src/ package
- Follow the established patterns from Phase 0 documentation
- All code should be tested with pytest

## Contributing

1. Check which phase you're implementing
2. Look for the corresponding module README
3. Follow existing patterns and code style
4. Add tests for new functionality
5. Update this README if adding new modules

## Learn More

- See [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) for the full roadmap
- See [../CLAUDE.md](../CLAUDE.md) for architecture guidance
- Each module has its own README with detailed documentation
