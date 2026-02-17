# Retrieval Module (Phase 4)

**Status:** Planning phase - to be implemented in Phase 4

**Purpose:** Vector database operations for semantic document search.

## Supported Vector Stores

| Store | Local | Scalable | Speed | Cost | Best For |
|-------|-------|----------|-------|------|----------|
| **Chroma** | ✅ Yes | ❌ No | Very Fast | Free | Development |
| **Milvus** | ❌ No | ✅ Yes | Fast | Free | Production |

## Design Pattern

All vector stores implement the same interface:

```python
class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, texts: List[str], embeddings: List[List[float]],
            metadata: List[Dict]) -> List[str]:
        """Store texts with embeddings"""
        pass

    @abstractmethod
    def search(self, embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Find similar documents"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Remove documents"""
        pass
```

## Key Files (to be implemented in Phase 4)

- `base.py` - Abstract vector store interface
- `chroma_store.py` - ChromaDB implementation (local)
- `milvus_store.py` - Milvus implementation (scalable)
- `factory.py` - Vector store selection factory
- `models.py` - Type definitions (SearchResult, etc.)

## How It Works

### Semantic Search Pipeline

```
User Query
    ↓
[Convert to embedding via LLM]
    ↓
[Search vector database]
    ↓
[Return top-k similar documents]
    ↓
[Pass to Agent for answer generation]
```

### Data Flow

```
Phase 5: Ingestion
  ├─ Load document
  ├─ Split into chunks
  ├─ Create embeddings (via LLM)
  └─ Store in Vector DB + metadata in Postgres

Phase 4: Retrieval (this module)
  ├─ Receive query
  ├─ Create query embedding
  ├─ Find similar vectors
  └─ Return matching documents
```

## Usage Examples (Phase 4+)

### Store Documents
```python
from src.retrieval.factory import get_vector_store

vector_store = get_vector_store()

# Add documents with embeddings
vector_store.add(
    texts=[
        "Our vacation policy provides 20 days annually",
        "Remote work is allowed 3 days per week"
    ],
    embeddings=[
        [0.1, 0.2, 0.3, ...],  # Embedding for first text
        [0.15, 0.25, 0.35, ...] # Embedding for second text
    ],
    metadata=[
        {"source": "hr_policy.pdf", "page": 3},
        {"source": "remote_work.docx", "page": 5}
    ]
)
```

### Search for Similar Documents
```python
# Create embedding for query
query = "What are the vacation days?"
query_embedding = llm.embed(query)  # From Phase 3 LLM module

# Find similar documents
results = vector_store.search(query_embedding, top_k=5)

# Results
for result in results:
    print(f"Score: {result.score:.2%}")
    print(f"Text: {result.text}")
    print(f"Source: {result.metadata['source']}")
```

### Delete Documents
```python
# Remove a document and all its chunks
vector_store.delete(["doc_id_1", "doc_id_2"])
```

## Metadata Management

Each vector is stored with metadata:
- `source`: Document filename
- `page`: Page number (for PDFs)
- `chunk_index`: Position in document
- `timestamp`: When document was ingested
- `custom_fields`: User-defined metadata

Metadata enables:
- Filtering by document source
- Tracking document versions
- User-specific access control
- Citation generation

## Configuration Dependencies

Relies on Phase 2 configuration:
- `settings.vector_store` - Which store (chroma/milvus)
- `settings.chroma_persist_directory` - Where to save local data
- `settings.milvus_host`, `settings.milvus_port` - Milvus connection

## Performance Considerations

### Chroma (Local)
- Stores in `.chroma_data/` directory
- Good for <1M documents
- Very fast (no network)
- Perfect for development

### Milvus (Production)
- Separate service (Docker container)
- Good for >1M documents
- Distributed indexing
- Advanced features (sharding, replication)

## Integration with Other Modules

- **Phase 3 (LLM):** Uses `llm.embed()` to create embeddings
- **Phase 5 (Ingestion):** Stores document embeddings
- **Phase 6 (Basic RAG):** Retrieves documents for answer generation
- **Phase 7 (Agents):** Retriever agent uses this module
- **Phase 11 (Evaluation):** Tracks retrieval quality metrics

## Testing (Phase 4+)

```python
import pytest
from src.retrieval.factory import get_vector_store

@pytest.fixture
def vector_store():
    return get_vector_store()

def test_add_and_search(vector_store):
    """Test adding and retrieving documents"""
    vector_store.add(
        texts=["test document"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadata=[{"source": "test.txt"}]
    )

    results = vector_store.search([0.1, 0.2, 0.3], top_k=1)
    assert len(results) == 1
    assert results[0].text == "test document"

def test_metadata_filtering(vector_store):
    """Test filtering by metadata"""
    vector_store.add(
        texts=["doc1", "doc2"],
        embeddings=[[0.1, 0.2], [0.15, 0.25]],
        metadata=[
            {"source": "a.pdf"},
            {"source": "b.pdf"}
        ]
    )

    # Should filter by source
    results = vector_store.search_with_filter(
        embedding=[0.1, 0.2],
        filters={"source": "a.pdf"}
    )
    assert results[0].metadata["source"] == "a.pdf"
```

## Future Enhancements

- [ ] Hybrid search (vector + keyword)
- [ ] Re-ranking of search results
- [ ] Document versioning and updates
- [ ] Distributed search across shards
- [ ] GPU acceleration for large-scale search
- [ ] Time-based filtering (search recent documents)
- [ ] User-specific search (multi-tenancy)
