"""Document Loaders

Load text content from various file formats.
"""

from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Base class for document loaders."""

    @staticmethod
    def load(file_path: str) -> Tuple[str, dict]:
        """Load document and extract text.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (extracted_text, metadata_dict)

        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFileTypeError: If file type not supported
            ExtractionError: If extraction fails
        """
        raise NotImplementedError


class TextFileLoader(DocumentLoader):
    """Loader for plain text files."""

    @staticmethod
    def load(file_path: str) -> Tuple[str, dict]:
        """Load plain text file.

        Args:
            file_path: Path to .txt file

        Returns:
            Tuple of (text_content, metadata)
        """
        from atlasrag.src.ingestion.exceptions import ExtractionError

        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = {
                "file_type": "txt",
                "file_size": path.stat().st_size,
                "encoding": "utf-8",
            }

            return content, metadata
        except Exception as e:
            raise ExtractionError(f"Failed to load text file: {str(e)}")


class PDFLoader(DocumentLoader):
    """Loader for PDF files."""

    @staticmethod
    def load(file_path: str) -> Tuple[str, dict]:
        """Load PDF file and extract text.

        Args:
            file_path: Path to .pdf file

        Returns:
            Tuple of (text_content, metadata)
        """
        from pypdf import PdfReader
        from atlasrag.src.ingestion.exceptions import ExtractionError

        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            reader = PdfReader(path)
            text_content = ""
            page_count = len(reader.pages)

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_content += f"\n--- Page {page_num} ---\n{text}"

            metadata = {
                "file_type": "pdf",
                "file_size": path.stat().st_size,
                "page_count": page_count,
                "author": reader.metadata.get("/Author", "") if reader.metadata else "",
                "title": reader.metadata.get("/Title", "") if reader.metadata else "",
            }

            return text_content, metadata
        except Exception as e:
            raise ExtractionError(f"Failed to load PDF file: {str(e)}")


class DocxLoader(DocumentLoader):
    """Loader for Word documents."""

    @staticmethod
    def load(file_path: str) -> Tuple[str, dict]:
        """Load DOCX file and extract text.

        Args:
            file_path: Path to .docx file

        Returns:
            Tuple of (text_content, metadata)
        """
        from docx import Document
        from atlasrag.src.ingestion.exceptions import ExtractionError

        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            doc = Document(path)
            text_content = "\n".join(paragraph.text for paragraph in doc.paragraphs)

            # Extract metadata
            props = doc.core_properties
            metadata = {
                "file_type": "docx",
                "file_size": path.stat().st_size,
                "author": props.author or "",
                "title": props.title or "",
                "subject": props.subject or "",
                "created": str(props.created) if props.created else "",
                "modified": str(props.modified) if props.modified else "",
            }

            return text_content, metadata
        except Exception as e:
            raise ExtractionError(f"Failed to load DOCX file: {str(e)}")


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files."""

    @staticmethod
    def load(file_path: str) -> Tuple[str, dict]:
        """Load Markdown file.

        Args:
            file_path: Path to .md file

        Returns:
            Tuple of (text_content, metadata)
        """
        from atlasrag.src.ingestion.exceptions import ExtractionError

        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = {
                "file_type": "md",
                "file_size": path.stat().st_size,
                "encoding": "utf-8",
            }

            return content, metadata
        except Exception as e:
            raise ExtractionError(f"Failed to load Markdown file: {str(e)}")


class HTMLLoader(DocumentLoader):
    """Loader for HTML files."""

    @staticmethod
    def load(file_path: str) -> Tuple[str, dict]:
        """Load HTML file and extract text.

        Args:
            file_path: Path to .html file

        Returns:
            Tuple of (text_content, metadata)
        """
        from bs4 import BeautifulSoup
        from atlasrag.src.ingestion.exceptions import ExtractionError

        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text_content = soup.get_text(separator="\n")

            metadata = {
                "file_type": "html",
                "file_size": path.stat().st_size,
                "title": soup.title.string if soup.title else "",
            }

            return text_content, metadata
        except Exception as e:
            raise ExtractionError(f"Failed to load HTML file: {str(e)}")


class UniversalLoader:
    """Universal loader that routes to appropriate loader based on file type."""

    LOADER_MAP = {
        ".txt": TextFileLoader,
        ".md": MarkdownLoader,
        ".pdf": PDFLoader,
        ".docx": DocxLoader,
        ".html": HTMLLoader,
    }

    @classmethod
    def load(cls, file_path: str) -> Tuple[str, dict]:
        """Load document from any supported file type.

        Args:
            file_path: Path to the document

        Returns:
            Tuple of (text_content, metadata)

        Raises:
            UnsupportedFileTypeError: If file type not supported
            FileNotFoundError: If file doesn't exist
            ExtractionError: If extraction fails
        """
        from atlasrag.src.ingestion.exceptions import (
            UnsupportedFileTypeError,
            FileNotFoundError as IngestionFileNotFoundError,
        )

        path = Path(file_path)

        if not path.exists():
            raise IngestionFileNotFoundError(f"File not found: {file_path}")

        file_ext = path.suffix.lower()

        if file_ext not in cls.LOADER_MAP:
            supported = ", ".join(cls.LOADER_MAP.keys())
            raise UnsupportedFileTypeError(
                f"File type '{file_ext}' not supported. "
                f"Supported types: {supported}"
            )

        loader = cls.LOADER_MAP[file_ext]
        return loader.load(file_path)
