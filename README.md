# AtlasRAG - Enterprise Agentic RAG Platform

Production-grade Retrieval-Augmented Generation (RAG) system with grounded, fact-checked answers.

## 🚀 Quick Start - Phase 0: Setup & Understanding

**New to this project?** Start here! Phase 0 takes about 20-30 minutes.

### Step 1: Verify Prerequisites

```bash
python3 verify_prerequisites.py
```

This script checks if you have:
- ✅ Python 3.11+
- ✅ Docker
- ✅ Ollama

If anything is missing, the script will provide platform-specific installation instructions.

### Step 2: Read the Documentation

1. **Concepts** - Understand the core technologies:
   ```bash
   cat docs/CONCEPTS.md
   ```
   Learn about Docker, Ollama, FastAPI, and Vector Databases in simple terms.

2. **Installation** - Install missing prerequisites:
   ```bash
   cat docs/INSTALLATION.md
   ```
   Platform-specific guides for macOS, Linux, and Windows.

3. **Phase 0 Checklist** - Complete the setup:
   ```bash
   cat docs/PHASE_0_CHECKLIST.md
   ```
   Actionable checklist with verification steps and reflection questions.

### Step 3: Test Your Environment

```bash
# Test Docker
docker run hello-world

# Pull first Ollama model (takes a few minutes)
ollama pull llama2

# List installed models
ollama list
```

## 📋 Project Status

| Phase | Status | Deliverables |
|-------|--------|---|
| **Phase 0** 🟢 | Starting | Prerequisites verification, concepts docs, installation guide |
| Phase 1 | Next | Project structure (atlasrag/src/, infra/, tests/) |
| Phase 2-3 | Planned | Configuration system, LLM abstraction layer |
| Phase 4-6 | Planned | Vector store, document ingestion, basic RAG |
| Phase 7-9 | Planned | LangGraph agents, verification loop, FastAPI API |
| Phase 10-15 | Planned | Observability, evaluation, Docker, Kubernetes, polish |

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for detailed roadmap.

## 🎯 What is AtlasRAG?

AtlasRAG enables organizations to ask questions about company documents with:

- **Grounded answers**: Every claim backed by source documents
- **Citations**: Know exactly where information came from
- **Verification**: Automatic fact-checking and repair loops
- **Flexibility**: Switch between Ollama (local), OpenAI, or Gemini with one config change
- **Privacy**: Works offline or on-premises using Ollama
- **Enterprise-ready**: Production deployment on Kubernetes

### Example

```
User: "What's our vacation policy?"

AtlasRAG: "Your company provides 20 days of vacation annually
(HR Policy, p. 3). Additional unpaid leave may be requested
(Benefits Guide, p. 5). Confidence: 94%"
```

## 🏗️ Architecture

```
FastAPI Server (Question Handler)
    ↓
LangGraph Agents (Planner, Retriever, Answerer, Verifier)
    ↓
Ollama LLM (Answer Generation)
Vector Database (Document Search)
    ↓
Response with Citations
```

## 🛠️ Tech Stack

- **Framework**: FastAPI (HTTP API)
- **Orchestration**: LangGraph (multi-agent coordination)
- **LLM Runtime**: Ollama (local) + OpenAI/Gemini (cloud)
- **Vector DB**: Chroma (dev) + Milvus (production)
- **Deployment**: Docker + Kubernetes
- **Observability**: OpenTelemetry + Prometheus

## 📚 Documentation

- [CLAUDE.md](CLAUDE.md) - Guidance for Claude Code
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) - Complete 15-phase roadmap
- [docs/CONCEPTS.md](docs/CONCEPTS.md) - Core technology concepts
- [docs/INSTALLATION.md](docs/INSTALLATION.md) - Installation guide
- [docs/PHASE_0_CHECKLIST.md](docs/PHASE_0_CHECKLIST.md) - Phase 0 completion checklist

## 🤔 Common Questions

**Q: Do I need to install everything to get started?**
A: Only Python 3.11+, Docker, and Ollama are required. The verification script will help you.

**Q: Can I use this offline?**
A: Yes! Phase 0 and beyond use Ollama by default, which runs completely offline.

**Q: Do I have enough disk space?**
A: Ollama models are ~4GB each. One model is fine for development.

**Q: When should I switch to OpenAI/Gemini?**
A: You can demo everything with Ollama first, then switch to cloud providers in production.

## 🎓 Learning Path

1. **Phase 0** ← You are here
   - Understand the tools
   - Set up your environment
   - Estimated time: 20-30 minutes

2. **Phase 1** (Next)
   - Create project structure
   - Set up Python dependencies
   - Create docker-compose.yml

3. **Phases 2-15**
   - Build the RAG system incrementally
   - Add agents, verification, APIs, deployment

Each phase builds on the previous one. No magic black boxes—understand everything!

## 🚀 Ready to Begin?

Make sure you've completed Phase 0:

```bash
# Run the verification script
python3 verify_prerequisites.py

# You should see: ✓ All prerequisites verified!
```

If you see green checkmarks, you're ready to proceed to Phase 1.

## 📖 Philosophy

> **"Build step-by-step, understand everything, no magic black boxes."**

This project teaches you how RAG systems work by building one from scratch. Every component has a purpose. Every phase builds understanding.

---

**Status**: Phase 0 - Setup & Understanding
**Current Activity**: Installing prerequisites and understanding core concepts
**Next Phase**: Phase 1 - Project Skeleton
