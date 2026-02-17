# FastAPI Server (Phase 9)

**Status:** Planning phase - to be implemented in Phase 9

**Purpose:** HTTP API for querying and document management.

## API Endpoints

### POST /query
Ask a question about company documents.

**Request:**
```json
{
  "query": "What's our vacation policy?",
  "top_k": 5,
  "timeout_seconds": 30
}
```

**Response:**
```json
{
  "answer": "Our vacation policy provides 20 days annually...",
  "citations": [
    {
      "text": "20 days annually",
      "source": "HR_Policy_2024.pdf",
      "page": 3
    }
  ],
  "confidence": 0.94,
  "processing_time_seconds": 2.3
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid request
- `503` - LLM provider unavailable

### POST /ingest
Upload a document to the knowledge base.

**Request:** Multipart form data with file
```
POST /ingest
Content-Type: multipart/form-data

file: <binary PDF/DOCX/etc>
metadata: {"department": "HR"}
```

**Response:**
```json
{
  "job_id": "uuid-1234",
  "status": "processing",
  "file_name": "policy.pdf"
}
```

### GET /status/{job_id}
Get document ingestion status.

**Response:**
```json
{
  "job_id": "uuid-1234",
  "status": "processing",
  "progress": 0.65,
  "chunks_processed": 15,
  "error": null
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-02-16T10:30:00Z",
  "services": {
    "database": "ok",
    "vector_store": "ok",
    "llm": "ok"
  }
}
```

### GET /metrics
Prometheus metrics in text format.

**Response:**
```
# HELP atlasrag_queries_total Total number of queries
# TYPE atlasrag_queries_total counter
atlasrag_queries_total 1234

# HELP atlasrag_query_latency_seconds Query processing latency
# TYPE atlasrag_query_latency_seconds histogram
atlasrag_query_latency_seconds_bucket{le="1.0"} 450
atlasrag_query_latency_seconds_bucket{le="5.0"} 1200
```

## Key Files (to be implemented in Phase 9)

- `main.py` - FastAPI app initialization
- `routes/query.py` - /query endpoint logic
- `routes/ingest.py` - /ingest endpoint logic
- `routes/health.py` - Health check endpoints
- `routes/status.py` - Job status tracking
- `models.py` - Pydantic request/response schemas
- `middleware.py` - CORS, auth, logging middleware
- `dependencies.py` - FastAPI dependency injection
- `exceptions.py` - Custom exceptions and error handlers

## Request/Response Models

```python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    timeout_seconds: int = 30

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
    processing_time_seconds: float

class Citation(BaseModel):
    text: str
    source: str
    page: Optional[int]
```

## Configuration Dependencies

Relies on Phase 2 configuration:
- `settings.api_host` - Bind address (default: 0.0.0.0)
- `settings.api_port` - Server port (default: 8000)
- `settings.api_workers` - Number of workers (default: 4)
- `settings.api_reload` - Hot-reload (default: true in dev)
- `settings.cors_allow_origins` - CORS origins
- `settings.cors_allow_credentials` - Allow credentials

## Integration with Other Modules

- **Phase 3 (LLM):** Uses `llm.generate()` for answer generation
- **Phase 4 (Retrieval):** Uses `vector_store.search()` for document lookup
- **Phase 5 (Ingestion):** Exposes `/ingest` endpoint
- **Phase 7 (Agents):** Routes queries through agent graph
- **Phase 10 (Observability):** Emits metrics

## Middleware

### Authentication Middleware
Optional API key authentication:
```python
# Client request
GET /query
X-API-Key: your-api-key
```

### CORS Middleware
Allow cross-origin requests from allowed origins.

### Logging Middleware
Log all requests and responses for debugging.

### Error Handling Middleware
Convert exceptions to proper HTTP responses.

## Usage Examples (Phase 9+)

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What's our vacation policy?"}
)

result = response.json()
print(result["answer"])
```

### JavaScript Client
```javascript
const response = await fetch("http://localhost:8000/query", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({query: "What's our vacation policy?"})
});

const result = await response.json();
console.log(result.answer);
```

### cURL
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What'\''s our vacation policy?"}'
```

## Error Handling

Errors return appropriate HTTP status codes:

```python
from fastapi import HTTPException

# 400 Bad Request
if not request.query:
    raise HTTPException(status_code=400, detail="Query required")

# 503 Service Unavailable
if not llm.is_available():
    raise HTTPException(status_code=503, detail="LLM provider offline")

# 500 Internal Server Error
try:
    # Process query
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

## Testing (Phase 9+)

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_query_endpoint():
    """Test /query endpoint"""
    response = client.post(
        "/query",
        json={"query": "What's our vacation policy?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data

def test_health_check():
    """Test /health endpoint"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_cors_headers():
    """Test CORS headers"""
    response = client.options("/query")

    assert "access-control-allow-origin" in response.headers
```

## Performance Optimization

- **Async endpoints:** All endpoints are async for concurrency
- **Connection pooling:** Reuse database/vector store connections
- **Caching:** Cache frequent queries
- **Rate limiting:** Prevent API abuse

## Deployment

### Docker
```bash
docker build -t atlasrag:api .
docker run -p 8000:8000 atlasrag:api
```

### Kubernetes
See Phase 14 for K8s manifests.

## Future Enhancements

- [ ] Streaming responses for long answers
- [ ] WebSocket support for real-time updates
- [ ] Bulk query API
- [ ] Advanced filtering
- [ ] Custom prompt templates
- [ ] Response caching
