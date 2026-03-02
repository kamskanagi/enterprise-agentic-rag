# Contributing to AtlasRAG

## Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/enterprise-agentic-rag.git
   cd enterprise-agentic-rag
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Start backing services**

   ```bash
   ollama pull llama2
   docker compose up -d postgres chroma
   ```

5. **Run the test suite**

   ```bash
   pytest atlasrag/tests/ -v
   ```

## Code Style

- **Formatter**: [Black](https://github.com/psf/black) (line length 88)
- **Linter**: [Ruff](https://github.com/astral-sh/ruff)
- **Type checker**: [Mypy](https://mypy-lang.org/) (strict mode on `src/`)

Run all checks:

```bash
black atlasrag/
ruff check atlasrag/
mypy atlasrag/src/
```

## Testing

- Place unit tests in `atlasrag/tests/unit/`
- Place integration tests in `atlasrag/tests/integration/`
- Run a specific test: `pytest atlasrag/tests/unit/test_chunkers.py -v`
- Run with coverage: `pytest atlasrag/tests/ --cov=atlasrag/src`

All tests must pass before merging.

## Pull Request Process

1. Create a feature branch from `main`
2. Make focused, small changes (aim for under 400 lines)
3. Use conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
4. Ensure all tests pass and linting is clean
5. Open a PR with a clear description of what changed and why
6. Request review from at least one maintainer
