"""Basic RAG pipeline - end-to-end question answering with retrieval."""

import logging
from typing import Optional

from atlasrag.src.llm.base import BaseLLMProvider
from atlasrag.src.llm.factory import get_llm_client
from atlasrag.src.rag.exceptions import NoDocumentsFoundError
from atlasrag.src.rag.models import RAGConfig, RAGResponse, SourceReference
from atlasrag.src.rag.prompts import DEFAULT_SYSTEM_PROMPT, build_rag_prompt
from atlasrag.src.retrieval.base import BaseVectorStore
from atlasrag.src.retrieval.factory import get_vector_store

logger = logging.getLogger(__name__)


class BasicRAGPipeline:
    """Simple linear RAG pipeline: embed query -> search -> build prompt -> generate answer.

    This is the non-agentic pipeline. Agent-based orchestration comes in Phase 7.
    """

    def __init__(
        self,
        llm: Optional[BaseLLMProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.vector_store = vector_store or get_vector_store()
        self.config = config or RAGConfig()

    def query(self, question: str) -> RAGResponse:
        """Execute the full RAG pipeline for a question.

        Args:
            question: The user's question.

        Returns:
            RAGResponse with answer, sources, and metadata.

        Raises:
            NoDocumentsFoundError: When no relevant documents are found.
        """
        logger.info("RAG query: %s", question)

        # 1. Embed the question
        embedding_response = self.llm.embed(question)
        query_embedding = embedding_response.embedding

        # 2. Search vector store
        search_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.config.top_k,
            similarity_threshold=self.config.similarity_threshold,
        )

        if not search_results.results:
            raise NoDocumentsFoundError(
                f"No documents found for query: {question}"
            )

        # 3. Convert search results to source references
        sources = [
            SourceReference(
                content=result.content,
                source=result.metadata.get("source", "unknown"),
                page=result.metadata.get("page"),
                chunk_index=result.metadata.get("chunk_index"),
                similarity_score=result.similarity_score,
            )
            for result in search_results.results
        ]

        # 4. Build prompt
        system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        rag_prompt = f"{system_prompt}\n\n{build_rag_prompt(question, sources)}"

        # 5. Generate answer
        llm_response = self.llm.generate(
            prompt=rag_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        logger.info(
            "RAG response generated: %d sources, model=%s",
            len(sources),
            llm_response.model,
        )

        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            query=question,
            model=llm_response.model,
            provider=llm_response.provider,
            retrieval_count=len(sources),
        )

    def is_ready(self) -> bool:
        """Check if the pipeline's LLM and vector store are available."""
        try:
            return self.llm.is_available() and self.vector_store.is_available()
        except Exception:
            return False
