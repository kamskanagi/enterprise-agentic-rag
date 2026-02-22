"""Document Chunking Strategies

Split documents into context-preserving chunks.
"""

from typing import List
from atlasrag.src.ingestion.models import DocumentChunk, ChunkingConfig
from atlasrag.src.ingestion.exceptions import ChunkingError


class DocumentChunker:
    """Split documents into overlapping chunks."""

    def __init__(self, config: ChunkingConfig = None):
        """Initialize chunker with configuration.

        Args:
            config: ChunkingConfig with chunk size and overlap
        """
        self.config = config or ChunkingConfig()

    def chunk(
        self,
        text: str,
        source_file: str,
        source_url: str = None,
        page_number: int = None,
        custom_metadata: dict = None,
    ) -> List[DocumentChunk]:
        """Split text into chunks with metadata.

        Args:
            text: Full document text
            source_file: Original filename
            source_url: Optional source URL
            page_number: Optional page number
            custom_metadata: Optional custom metadata

        Returns:
            List of DocumentChunk objects

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            if not text or len(text) == 0:
                raise ValueError("Cannot chunk empty text")

            chunks = []
            chunk_size = self.config.chunk_size
            overlap = self.config.chunk_overlap
            separator = self.config.separator

            # Handle empty separator (character-level chunking)
            if not separator:
                return self._chunk_by_character(
                    text, source_file, source_url, page_number, custom_metadata
                )

            # Split by separator first for context preservation
            sections = text.split(separator)

            current_chunk = ""
            current_offset = 0
            chunk_index = 0

            for section in sections:
                # Add separator back if needed
                if self.config.keep_separator and current_chunk:
                    section_with_sep = separator + section
                else:
                    section_with_sep = section

                # If adding this section would exceed chunk_size
                if len(current_chunk + section_with_sep) > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_start = current_offset
                    chunks.append(
                        DocumentChunk(
                            content=current_chunk.strip(),
                            chunk_index=chunk_index,
                            source_file=source_file,
                            source_url=source_url,
                            page_number=page_number,
                            start_offset=chunk_start,
                            end_offset=current_offset + len(current_chunk),
                            custom_metadata=custom_metadata or {},
                        )
                    )

                    chunk_index += 1

                    # Keep overlap from end of previous chunk
                    if overlap > 0 and len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + section_with_sep
                        current_offset += len(current_chunk) - len(overlap_text) - len(
                            section_with_sep
                        )
                    else:
                        current_chunk = section_with_sep
                        current_offset += len(current_chunk)
                else:
                    # Add section to current chunk
                    current_chunk += section_with_sep
                    current_offset += len(section_with_sep)

            # Add final chunk
            if current_chunk.strip():
                chunks.append(
                    DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        source_file=source_file,
                        source_url=source_url,
                        page_number=page_number,
                        start_offset=current_offset - len(current_chunk),
                        end_offset=current_offset,
                        custom_metadata=custom_metadata or {},
                    )
                )

            return chunks
        except Exception as e:
            raise ChunkingError(f"Document chunking failed: {str(e)}")

    def _chunk_by_character(
        self,
        text: str,
        source_file: str,
        source_url: str = None,
        page_number: int = None,
        custom_metadata: dict = None,
    ) -> List[DocumentChunk]:
        """Chunk text by character (fallback strategy).

        Args:
            text: Full document text
            source_file: Original filename
            source_url: Optional source URL
            page_number: Optional page number
            custom_metadata: Optional custom metadata

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        for i in range(0, len(text), chunk_size - overlap):
            end = min(i + chunk_size, len(text))
            chunk_text = text[i:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        chunk_index=len(chunks),
                        source_file=source_file,
                        source_url=source_url,
                        page_number=page_number,
                        start_offset=i,
                        end_offset=end,
                        custom_metadata=custom_metadata or {},
                    )
                )

        return chunks


class RecursiveChunker(DocumentChunker):
    """Chunk using recursive separator strategy.

    Tries separators in order: paragraph, sentence, word, character.
    """

    def __init__(self, config: ChunkingConfig = None):
        """Initialize recursive chunker.

        Args:
            config: ChunkingConfig (uses default separators)
        """
        if config is None:
            config = ChunkingConfig(separator="\n\n")  # Paragraph by default
        super().__init__(config)

    def chunk(
        self,
        text: str,
        source_file: str,
        source_url: str = None,
        page_number: int = None,
        custom_metadata: dict = None,
    ) -> List[DocumentChunk]:
        """Chunk using recursive separators.

        Args:
            text: Full document text
            source_file: Original filename
            source_url: Optional source URL
            page_number: Optional page number
            custom_metadata: Optional custom metadata

        Returns:
            List of DocumentChunk objects
        """
        separators = ["\n\n", "\n", ". ", " "]

        for separator in separators:
            # Try this separator
            self.config.separator = separator
            try:
                chunks = super().chunk(
                    text, source_file, source_url, page_number, custom_metadata
                )

                # Check if chunks are reasonably sized
                if len(chunks) > 0:
                    avg_size = sum(len(c.content) for c in chunks) / len(chunks)
                    if avg_size >= self.config.chunk_size * 0.5:
                        # Good enough, use these chunks
                        return chunks
            except ChunkingError:
                continue

        # Fallback to character-level chunking
        return self._chunk_by_character(
            text, source_file, source_url, page_number, custom_metadata
        )


class SentenceChunker(DocumentChunker):
    """Chunk by sentences to preserve meaning."""

    def chunk(
        self,
        text: str,
        source_file: str,
        source_url: str = None,
        page_number: int = None,
        custom_metadata: dict = None,
    ) -> List[DocumentChunk]:
        """Chunk by sentences.

        Args:
            text: Full document text
            source_file: Original filename
            source_url: Optional source URL
            page_number: Optional page number
            custom_metadata: Optional custom metadata

        Returns:
            List of DocumentChunk objects
        """
        import re

        # Split by sentence-ending punctuation
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_chunk = ""
        current_offset = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add space if not first sentence in chunk
            if current_chunk:
                test_chunk = current_chunk + " " + sentence
            else:
                test_chunk = sentence

            if len(test_chunk) <= self.config.chunk_size or not current_chunk:
                current_chunk = test_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(
                        DocumentChunk(
                            content=current_chunk.strip(),
                            chunk_index=chunk_index,
                            source_file=source_file,
                            source_url=source_url,
                            page_number=page_number,
                            start_offset=current_offset - len(current_chunk),
                            end_offset=current_offset,
                            custom_metadata=custom_metadata or {},
                        )
                    )
                    chunk_index += 1

                current_chunk = sentence

            current_offset += len(sentence) + 1

        # Add final chunk
        if current_chunk:
            chunks.append(
                DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    source_file=source_file,
                    source_url=source_url,
                    page_number=page_number,
                    start_offset=current_offset - len(current_chunk),
                    end_offset=current_offset,
                    custom_metadata=custom_metadata or {},
                )
            )

        return chunks
