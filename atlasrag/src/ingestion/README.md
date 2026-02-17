# Document Ingestion Pipeline (Phase 5)

**Status:** Planning phase - to be implemented in Phase 5

**Purpose:** Process raw documents into searchable knowledge.

## Pipeline Stages

```
Raw Document (PDF, DOCX, HTML, etc.)
    ↓
[Stage 1: Load] Extract text content
    ↓
[Stage 2: Clean] Remove artifacts, normalize text
    ↓
[Stage 3: Chunk] Split into context-preserving segments
    ↓
[Stage 4: Embed] Convert chunks to vectors
    ↓
[Stage 5: Store] Save to Vector DB + Postgres
    ↓
✅ Ready for Retrieval
```

## Supported File Formats

| Format | Extension | Library | Notes |
|--------|-----------|---------|-------|
| PDF | `.pdf` | pypdf | Handles tables, images |
| Word | `.docx` | python-docx | Preserves formatting |
| Plain Text | `.txt` | Built-in | Simplest format |
| HTML | `.html` | beautifulsoup4 | Strips markup, extracts content |
| Markdown | `.md` | Built-in | Treats as plain text |

## Key Files (to be implemented in Phase 5)

- `loaders.py` - Document loaders for each file type
- `cleaners.py` - Text preprocessing and normalization
- `chunkers.py` - Smart text splitting strategies
- `pipeline.py` - Orchestrates the full ingestion flow
- `models.py` - Database models for ingestion jobs
- `exceptions.py` - Custom exceptions (LoaderError, etc.)

## Chunking Strategy

### The Challenge
- Too small chunks: Lost context, bad embeddings
- Too large chunks: Single chunk may not match query
- Wrong split point: Breaks semantic meaning

### The Solution: Intelligent Chunking
```
Original: "The company provides vacation days. Employees get 20 days annually. ..."

Chunk 1: "The company provides vacation days. Employees get 20 days annually."
         (1000 chars with 200-char overlap to next chunk)

Chunk 2: "Employees get 20 days annually. Additional unpaid leave may be requested."
         (200-char overlap from previous, 200-char overlap to next)

Chunk 3: "Additional unpaid leave may be requested. No more than 40 days total per year."
```

### Configuration (from .env)
- `CHUNK_SIZE=1000` - Characters per chunk
- `CHUNK_OVERLAP=200` - Overlap between chunks (for context)

## Usage Examples (Phase 5+)

### Ingest a Single Document
```python
from src.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

# Upload and process document
job_id = pipeline.ingest_file(
    file_path="/path/to/document.pdf",
    metadata={"department": "HR", "year": 2024}
)

# Check status
status = pipeline.get_status(job_id)
print(f"Status: {status.status}")  # pending, processing, completed, failed
print(f"Progress: {status.progress:.0%}")
print(f"Chunks processed: {status.chunks_processed}")
```

### Ingest from Directory
```python
# Process all documents in a directory
job_ids = pipeline.ingest_directory(
    directory_path="/path/to/documents/",
    recursive=True
)

# Check overall progress
for job_id in job_ids:
    status = pipeline.get_status(job_id)
    print(f"{job_id}: {status.progress:.0%}")
```

### Track Ingestion
```python
# Listen to ingestion events
from src.ingestion.pipeline import IngestionEvent

def on_chunk_processed(event: IngestionEvent):
    print(f"Processed chunk: {event.chunk_index}")

pipeline.on_chunk_processed(on_chunk_processed)
```

## Configuration Dependencies

Relies on Phase 2 configuration:
- `settings.chunk_size` - Characters per chunk (default: 1000)
- `settings.chunk_overlap` - Overlap between chunks (default: 200)
- `settings.max_document_size_mb` - Max file size (default: 50MB)
- `settings.supported_file_types` - Allowed formats

Requires Phase 3-4:
- `settings.vector_store` - Where to store embeddings
- `settings.database_url` - PostgreSQL for metadata

## Integration with Other Modules

- **Phase 3 (LLM):** Uses `llm.embed()` to create embeddings
- **Phase 4 (Retrieval):** Stores chunks in vector DB
- **Phase 6 (Basic RAG):** Uses ingested documents for retrieval
- **Phase 9 (API):** Exposes via `/ingest` endpoint

## Error Handling

```python
from src.ingestion.exceptions import (
    LoaderError,        # File format not supported
    ChunkingError,      # Failed to split document
    EmbeddingError,     # Failed to create embedding
    StorageError        # Failed to persist
)

try:
    pipeline.ingest_file("document.pdf")
except LoaderError as e:
    print(f"Unsupported file format: {e}")
except EmbeddingError as e:
    print(f"LLM provider offline: {e}")
```

## Database Schema (to be created in Phase 5)

```sql
CREATE TABLE ingestion_jobs (
    id UUID PRIMARY KEY,
    file_name VARCHAR(255),
    file_size_bytes BIGINT,
    status VARCHAR(50),  -- pending, processing, completed, failed
    progress FLOAT,      -- 0.0 to 1.0
    chunks_processed INTEGER,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE documents (
    id UUID PRIMARY KEY,
    job_id UUID REFERENCES ingestion_jobs(id),
    file_name VARCHAR(255),
    file_type VARCHAR(50),
    content_hash VARCHAR(64),  -- Prevent duplicates
    metadata JSONB,
    created_at TIMESTAMP
);

CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    chunk_number INTEGER,
    content TEXT,
    embedding FLOAT8[],
    created_at TIMESTAMP
);
```

## Testing (Phase 5+)

```python
import pytest
from src.ingestion.pipeline import IngestionPipeline

@pytest.fixture
def pipeline():
    return IngestionPipeline()

def test_ingest_pdf(pipeline, tmp_path):
    """Test PDF ingestion"""
    # Create test PDF
    pdf_path = tmp_path / "test.pdf"
    # ... create PDF content ...

    job_id = pipeline.ingest_file(str(pdf_path))
    status = pipeline.get_status(job_id)

    assert status.status == "completed"
    assert status.chunks_processed > 0

def test_chunk_overlap(pipeline):
    """Test that chunks have proper overlap"""
    chunks = pipeline._chunk_text("This is chunk 1. " * 100)

    # Verify overlap
    for i in range(len(chunks) - 1):
        assert chunks[i][-100:] == chunks[i+1][:100]
```

## Performance Tuning

- **Memory:** Stream processing for large files
- **Speed:** Parallel embedding generation
- **Quality:** Optimize chunk size based on document type

## Future Enhancements

- [ ] Table and image extraction (OCR)
- [ ] Layout-preserving chunking
- [ ] Automatic metadata extraction (author, date, etc.)
- [ ] Duplicate detection (content hash)
- [ ] Incremental updates (don't re-process entire doc)
- [ ] Multi-language support
- [ ] Batch processing API
