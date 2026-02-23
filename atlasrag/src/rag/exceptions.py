"""Exceptions for the RAG pipeline."""


class RAGException(Exception):
    """Base exception for RAG pipeline errors."""

    pass


class NoDocumentsFoundError(RAGException):
    """Raised when vector store returns no results for a query."""

    pass


class ContextBuildError(RAGException):
    """Raised when prompt construction fails."""

    pass
