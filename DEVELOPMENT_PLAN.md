# AtlasRAG Development Plan
## 🎯 Partner Development Roadmap

**Philosophy:** Build step-by-step, understand everything, no magic black boxes.

---

## Overview: What Are We Building?

Imagine you have a big library of company documents (HR policies, security guides, etc.).
You want to ask questions like: *"What's our vacation policy?"*

**The Problem:**
- Regular AI might make stuff up (hallucinate)
- You need PROOF of where the answer came from
- You need it to work offline (privacy!) but also scale to cloud

**Our Solution (AtlasRAG):**
```
Documents → Chop into pieces → Store in searchable database
     ↓
Question → Find relevant pieces → Write answer WITH citations → VERIFY it's accurate
     ↓
If verification fails → Try again (repair loop)
```

---

## The Phases

### 🟢 PHASE 0: Setup & Understanding (Day 1)
**Goal:** Get our tools ready and understand the big picture

**What we'll do:**
- [ ] Install prerequisites (Python, Docker, Ollama)
- [ ] Create project folder structure
- [ ] Understand what each tool does (in simple terms)

**Concepts explained like you're 5:**
- **Docker** = A lunchbox that keeps your app and all its food (dependencies) together
- **Ollama** = A smart robot that lives on YOUR computer (not in the cloud)
- **FastAPI** = A waiter that takes orders (requests) and brings back food (responses)
- **Vector Database** = A library where books are organized by "vibes" not just titles

---

### 🟡 PHASE 1: Project Skeleton (Day 1-2)
**Goal:** Create all the folders and empty files — the "bones" of our project

**What we'll build:**
```
atlasrag/
├── src/
│   ├── api/           # The waiter (FastAPI)
│   ├── agents/        # The workers (Planner, Retriever, etc.)
│   ├── config/        # The settings (which LLM to use, etc.)
│   ├── ingestion/     # The librarian (processes documents)
│   ├── retrieval/     # The search engine
│   ├── llm/           # The brain (talks to Ollama/OpenAI/Gemini)
│   └── evaluation/    # The quality checker
├── infra/
│   ├── docker/        # The lunchbox recipes
│   └── k8s/           # The factory blueprints (Kubernetes)
├── eval/              # Test questions and answers
├── scripts/           # Helper tools
└── tests/             # Making sure nothing breaks
```

**Deliverables:**
- [ ] Full folder structure
- [ ] `pyproject.toml` (our shopping list of Python packages)
- [ ] `.env.example` (settings template)
- [ ] Basic `docker-compose.yml`

---

### 🟡 PHASE 2: Configuration System (Day 2-3)
**Goal:** Build the "settings panel" that lets us switch between Ollama/OpenAI/Gemini

**Why this matters:**
> "I can demo locally with Ollama, then flip ONE setting to use GPT-4 in production"

**What we'll build:**
- [ ] Settings loader (reads `.env` file)
- [ ] Provider selector (picks the right LLM based on settings)
- [ ] A simple test to prove switching works

**Concepts explained like you're 5:**
- **Environment Variables** = Sticky notes that tell your app how to behave
- **Provider Pattern** = Like having different TV remotes that all have a "power" button

---

### 🟡 PHASE 3: LLM Abstraction Layer (Day 3-4)
**Goal:** Create ONE way to talk to ANY LLM (Ollama, OpenAI, or Gemini)

**The magic:**
```python
# Same code works for ALL providers!
llm = get_llm_client()  # Picks based on settings
response = llm.generate("What is 2+2?")
embeddings = llm.embed("Hello world")
```

**What we'll build:**
- [ ] Base LLM interface (the "contract" all providers follow)
- [ ] Ollama provider
- [ ] OpenAI provider
- [ ] Gemini provider
- [ ] Provider factory (picks the right one)

---

### 🟡 PHASE 4: Vector Store Layer (Day 4-5)
**Goal:** Set up where we store document "vibes" (embeddings)

**What's an embedding?**
> Imagine turning a sentence into a point on a map.
> Similar sentences are CLOSE together on the map.
> "I love dogs" and "I adore puppies" are neighbors.
> "I love dogs" and "Tax regulations" are far apart.

**What we'll build:**
- [ ] Chroma integration (local, fast, easy)
- [ ] Milvus integration (optional, for "scale story")
- [ ] Same interface for both (just like LLM layer)

---

### 🟡 PHASE 5: Document Ingestion Pipeline (Day 5-7)
**Goal:** Turn messy documents into searchable knowledge

**The journey of a document:**
```
PDF/DOCX/HTML
    ↓ Extract text
Raw Text
    ↓ Clean (remove junk, fix formatting)
Clean Text
    ↓ Chunk (split into bite-sized pieces)
Chunks (with metadata: source, page, etc.)
    ↓ Embed (turn into "vibes")
Vectors + Metadata
    ↓ Store
Ready to search! 🎉
```

**What we'll build:**
- [ ] Document loaders (PDF, DOCX, HTML)
- [ ] Text cleaner
- [ ] Smart chunker (keeps context together)
- [ ] Embedding pipeline
- [ ] Storage in vector DB + Postgres

---

### 🟠 PHASE 6: Basic RAG (No Agents Yet) (Day 7-8)
**Goal:** Get a simple "ask question → get answer" working

**Simple flow:**
```
Question → Embed → Find similar chunks → Send to LLM → Get answer
```

**What we'll build:**
- [ ] Query embedding
- [ ] Similarity search
- [ ] Basic prompt template
- [ ] Simple answer generation

**Why start simple?**
> Before building a race car, make sure the wheels work!

---

### 🟠 PHASE 7: Agent Architecture with LangGraph (Day 8-12)
**Goal:** Build our team of AI workers

**Meet the team:**

| Agent | Job | Like... |
|-------|-----|---------|
| **Planner** | Breaks complex questions into simple ones | A manager splitting tasks |
| **Retriever** | Finds relevant document chunks | A librarian |
| **Answerer** | Writes the response with citations | A research assistant |
| **Verifier** | Checks if answer is grounded in sources | A fact-checker |

**What we'll build:**
- [ ] LangGraph setup
- [ ] Planner agent
- [ ] Retriever agent
- [ ] Answerer agent
- [ ] Verifier agent
- [ ] Graph that connects them

---

### 🟠 PHASE 8: Verification & Repair Loop (Day 12-14)
**Goal:** Make answers TRUSTWORTHY

**The verification rules:**
1. Every paragraph MUST have at least 1 citation
2. If a claim has no supporting source → flag as "unsupported"
3. If sources contradict each other → say "I'm not sure"

**The repair loop:**
```
Answer fails verification?
    ↓
Retrieve MORE sources
    ↓
Rewrite answer
    ↓
Verify again
    ↓
(Max 2 attempts, then give best effort with warnings)
```

**What we'll build:**
- [ ] Citation extractor
- [ ] Coverage checker
- [ ] Contradiction detector
- [ ] Repair loop logic

---

### 🟠 PHASE 9: FastAPI Server (Day 14-16)
**Goal:** Make our system accessible via HTTP

**Endpoints we'll create:**
```
POST /query          → Ask a question
POST /ingest         → Upload a document
GET  /health         → Is the system alive?
GET  /metrics        → Prometheus metrics
```

**What we'll build:**
- [ ] FastAPI app structure
- [ ] Request/response models
- [ ] Error handling
- [ ] Background tasks (async ingestion)

---

### 🔴 PHASE 10: Observability (Day 16-18)
**Goal:** See what's happening inside our system

**The three pillars:**
1. **Logs** = Diary entries ("User asked X at 3pm")
2. **Metrics** = Numbers over time (requests/second, latency)
3. **Traces** = Following one request through all steps

**What we'll build:**
- [ ] OpenTelemetry integration
- [ ] Prometheus metrics
- [ ] Grafana dashboard
- [ ] Structured logging

---

### 🔴 PHASE 11: Evaluation Suite (Day 18-21)
**Goal:** PROVE our system works with NUMBERS

**Our metrics:**
| Metric | What it measures | Target |
|--------|-----------------|--------|
| Recall@5 | Do we find the right docs? | >80% |
| Faithfulness | Is the answer grounded? | >90% |
| Citation Coverage | % sentences with citations | >95% |
| p95 Latency | Speed (worst case) | <5s |

**What we'll build:**
- [ ] Golden dataset (50-150 Q&A pairs)
- [ ] RAGAS integration
- [ ] Custom metrics
- [ ] Evaluation runner
- [ ] Baseline tracking

---

### 🔴 PHASE 12: CI/CD Quality Gates (Day 21-23)
**Goal:** Automatically catch regressions

**The rules:**
- ❌ Fail CI if Faithfulness drops >3%
- ❌ Fail CI if Recall@5 drops >5%  
- ❌ Fail CI if p95 latency increases >20%

**What we'll build:**
- [ ] GitHub Actions workflow
- [ ] Evaluation in CI
- [ ] Baseline comparison
- [ ] Slack/email alerts (optional)

---

### 🔴 PHASE 13: Docker & Local Demo (Day 23-25)
**Goal:** One command to run EVERYTHING locally

```bash
docker compose up
# That's it! Full system running.
```

**What we'll build:**
- [ ] Production-like docker-compose.yml
- [ ] Health checks
- [ ] Volume mounts for persistence
- [ ] Sample documents for demo

---

### 🔴 PHASE 14: Kubernetes Manifests (Day 25-27)
**Goal:** Show we can deploy to "real" production

**What we'll create:**
- [ ] Deployments (API, Worker)
- [ ] Services
- [ ] Ingress
- [ ] ConfigMaps & Secrets
- [ ] HPA (auto-scaling)
- [ ] ServiceMonitor (Prometheus scraping)

---

### 🟣 PHASE 15: Polish & Documentation (Day 27-30)
**Goal:** Make it portfolio-ready

- [ ] README with GIFs/screenshots
- [ ] Architecture diagram
- [ ] API documentation
- [ ] "How to contribute" guide
- [ ] Sample interview talking points

---

## How We'll Work Together

Each phase follows this pattern:

### 1️⃣ Explain
I explain what we're building and WHY (like you're 5)

### 2️⃣ Plan  
We outline the files and code structure

### 3️⃣ Build
We write the code together (you can use Claude Code CLI)

### 4️⃣ Test
We verify it works

### 5️⃣ Reflect
We summarize what we learned (good for interviews!)

---

## Ready to Start?

**Next step:** Phase 0 - Let's make sure your machine has everything installed!

Commands you'll need:
```bash
# Check Python
python --version  # Need 3.11+

# Check Docker
docker --version

# Check Ollama
ollama --version

# If missing, we'll install them together!
```

---

## Progress Tracker

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| 0 - Setup | ✅ Complete | Day 1 | Day 1 |
| 1 - Skeleton | ✅ Complete | Day 1 | Day 1 |
| 2 - Config | ✅ Complete | Day 1 | Day 1 |
| 3 - LLM Layer | ✅ Complete | Day 1 | Day 1 |
| 4 - Vector Store | ⬜ Next Up | | |
| 5 - Ingestion | ⬜ Not Started | | |
| 6 - Basic RAG | ⬜ Not Started | | |
| 7 - Agents | ⬜ Not Started | | |
| 8 - Verification | ⬜ Not Started | | |
| 9 - API | ⬜ Not Started | | |
| 10 - Observability | ⬜ Not Started | | |
| 11 - Evaluation | ⬜ Not Started | | |
| 12 - CI/CD | ⬜ Not Started | | |
| 13 - Docker | 🟡 Partial | Day 1 | |
| 14 - Kubernetes | ⬜ Not Started | | |
| 15 - Polish | ⬜ Not Started | | |

---

*Let's build something amazing! 🚀*
