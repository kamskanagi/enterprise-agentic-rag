"""Text Cleaning and Normalization

Preprocess extracted text for chunking and embedding.
"""

import re
import unicodedata
import logging

from atlasrag.src.ingestion.models import CleaningConfig
from atlasrag.src.ingestion.exceptions import CleaningError

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and normalize extracted text."""

    def __init__(self, config: CleaningConfig = None):
        """Initialize cleaner with configuration.

        Args:
            config: CleaningConfig with cleaning options
        """
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """Clean text according to configuration.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text

        Raises:
            CleaningError: If cleaning fails
        """
        try:
            # Normalize Unicode
            if self.config.normalize_unicode:
                text = self._normalize_unicode(text)

            # Remove extra whitespace
            if self.config.remove_extra_whitespace:
                text = self._remove_extra_whitespace(text)

            # Remove special characters
            if self.config.remove_special_characters:
                text = self._remove_special_characters(text)

            # Convert to lowercase
            if self.config.lowercase:
                text = text.lower()

            return text
        except Exception as e:
            raise CleaningError(f"Text cleaning failed: {str(e)}")

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize Unicode characters to NFC form.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_extra_whitespace(text: str) -> str:
        """Collapse multiple spaces and normalize line breaks.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse multiple newlines
        text = re.sub(r"\n{2,}", "\n", text)

        # Collapse multiple spaces on same line
        text = re.sub(r" {2,}", " ", text)

        # Strip leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines)

    @staticmethod
    def _remove_special_characters(text: str) -> str:
        """Remove special characters while keeping alphanumeric and basic punctuation.

        Args:
            text: Input text

        Returns:
            Text with special characters removed
        """
        # Keep alphanumeric, spaces, common punctuation, and newlines
        pattern = r"[^a-zA-Z0-9\s.,!?\-'\";\n]"
        return re.sub(pattern, "", text)

    @staticmethod
    def remove_headers_footers(text: str, lines_to_remove: int = 5) -> str:
        """Remove common headers and footers (first/last N lines).

        Args:
            text: Input text
            lines_to_remove: Number of lines to remove from start/end

        Returns:
            Text with headers/footers removed
        """
        lines = text.split("\n")

        if len(lines) <= 2 * lines_to_remove:
            return text  # Don't remove if too short

        # Remove first and last N lines
        lines = lines[lines_to_remove:-lines_to_remove]
        return "\n".join(lines)

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text.

        Args:
            text: Input text

        Returns:
            Text with URLs removed
        """
        url_pattern = r"https?://[^\s]+"
        return re.sub(url_pattern, "", text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text.

        Args:
            text: Input text

        Returns:
            Text with emails removed
        """
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        return re.sub(email_pattern, "", text)


class DocumentCleaner:
    """Higher-level document cleaning with multiple strategies."""

    @staticmethod
    def clean_for_chunking(
        text: str,
        remove_headers: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
    ) -> str:
        """Clean text optimized for chunking.

        Args:
            text: Input text
            remove_headers: Whether to remove headers/footers
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses

        Returns:
            Cleaned text
        """
        cleaner = TextCleaner()

        # Basic cleaning
        text = cleaner.clean(text)

        # Advanced cleaning
        if remove_headers:
            text = cleaner.remove_headers_footers(text)

        if remove_urls:
            text = cleaner.remove_urls(text)

        if remove_emails:
            text = cleaner.remove_emails(text)

        return text
