"""
Soak text importer — ingests sentences from text files and epubs.

Turns raw text into a pool of SentenceRecords that the SentencePool
can score and select from for high-density soak-in sessions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import IntensiveMorphConfig
from .models import SentenceRecord, IntensiveMorphDB
from .morphemizer import Morphemizer


class TextImporter:
    """
    Imports text from various sources and populates the sentence pool.
    """

    def __init__(self, config: IntensiveMorphConfig, db: IntensiveMorphDB, morphemizer: Morphemizer):
        self.config = config
        self.db = db
        self.morphemizer = morphemizer

    def import_text(self, text: str, source: str = "",
                    max_sentences: int = 10000) -> int:
        """
        Import raw text — split into sentences, lemmatize, store.

        Args:
            text: Raw text content.
            source: Source identifier (filename, book title, etc.)
            max_sentences: Maximum number of sentences to import.

        Returns:
            Number of sentences imported.
        """
        sentences = self._split_sentences(text)
        return self._import_sentences(sentences, source, max_sentences)

    def import_file(self, path: str | Path, max_sentences: int = 10000) -> int:
        """
        Import sentences from a text file.

        Args:
            path: Path to .txt file.
            max_sentences: Maximum sentences to import.

        Returns:
            Number of sentences imported.
        """
        path = Path(path)
        if not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.import_text(text, source=path.name, max_sentences=max_sentences)

    def import_epub(self, path: str | Path, max_sentences: int = 10000) -> int:
        """
        Extract and import sentences from an EPUB file.

        Args:
            path: Path to .epub file.
            max_sentences: Maximum sentences to import.

        Returns:
            Number of sentences imported.
        """
        path = Path(path)
        if not path.exists():
            return 0

        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise ImportError(
                "ebooklib is required for EPUB import. Install: pip install ebooklib"
            )

        book = epub.read_epub(str(path))
        text_parts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode("utf-8", errors="replace")
                # Strip HTML tags
                text = re.sub(r"<[^>]+>", " ", content)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    text_parts.append(text)

        full_text = "\n".join(text_parts)
        return self.import_text(full_text, source=path.name, max_sentences=max_sentences)

    def import_json(self, path: str | Path, text_key: str = "text",
                    translation_key: str = "translation",
                    max_sentences: int = 10000) -> int:
        """
        Import sentences from a JSON file.

        Expected format (two variants):
        - List of strings: ["sentence1", "sentence2", ...]
        - List of objects: [{"text": "sentence", "translation": "..."}, ...]

        Args:
            path: Path to JSON file.
            text_key: Key for sentence text in object format.
            translation_key: Key for translation in object format.
            max_sentences: Maximum sentences to import.

        Returns:
            Number of sentences imported.
        """
        path = Path(path)
        if not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            sentences = []
            translations = {}
            for item in data:
                if isinstance(item, str):
                    sentences.append(item)
                elif isinstance(item, dict):
                    text = item.get(text_key, "")
                    if text:
                        sentences.append(text)
                        if translation_key in item:
                            translations[text] = item[translation_key]

            count = 0
            for i, sentence in enumerate(sentences):
                if i >= max_sentences:
                    break
                lemmas = self.morphemizer.lemmatize(sentence)
                record = SentenceRecord(
                    text=sentence,
                    translation=translations.get(sentence, ""),
                    lemmas=lemmas,
                    source=path.name,
                    position=i,
                )
                if self.db.add_sentence(record):
                    count += 1
            return count

        return 0

    def import_sentence_list(self, sentences: List[str],
                             translations: Optional[Dict[str, str]] = None,
                             source: str = "") -> int:
        """Import a pre-split list of sentences."""
        if translations is None:
            translations = {}
        count = 0
        for i, sentence in enumerate(sentences):
            lemmas = self.morphemizer.lemmatize(sentence)
            record = SentenceRecord(
                text=sentence,
                translation=translations.get(sentence, ""),
                lemmas=lemmas,
                source=source,
                position=i,
            )
            if self.db.add_sentence(record):
                count += 1
        return count

    # --- Internal ---

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.
        Handles Russian punctuation (including ... and ?! markers).
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove obvious non-sentence content
        text = re.sub(r"[■□●○◆◇☆★♪♫♬¡¿]", " ", text)

        # Split on sentence-ending punctuation, keeping the delimiter
        # Russian uses the same .!? as English
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"'])", text)

        # Also handle ellipsis ...
        raw_sentences2 = []
        for s in raw_sentences:
            parts = re.split(r"(?<=\.\.\.)\s+(?=[А-ЯЁA-Z«\"'])", s)
            raw_sentences2.extend(parts)

        # Clean up
        result = []
        for s in raw_sentences2:
            s = s.strip()
            # Remove leading dialogue markers and quotes
            s = re.sub(r"^[-—–\s]+", "", s)
            s = re.sub(r'^[«"„″\']+', "", s)
            s = re.sub(r'[»"″\']+$', "", s)
            s = s.strip()

            # Filter
            if not s:
                continue
            if len(s) < 3:  # Too short
                continue
            if len(s) > 2000:  # Very long — skip (likely a formatting error)
                continue
            if s[0].islower():  # Doesn't start with capital letter
                continue

            result.append(s)

        return result

    def _import_sentences(self, raw_sentences: List[str],
                          source: str, max_sentences: int) -> int:
        """Batch-import a list of sentence strings with position tracking."""
        count = 0
        for i, sentence in enumerate(raw_sentences):
            if i >= max_sentences:
                break
            lemmas = self.morphemizer.lemmatize(sentence)
            if len(lemmas) < 2:
                continue  # Skip single-word "sentences"
            record = SentenceRecord(
                text=sentence,
                lemmas=lemmas,
                source=source,
                position=i,
            )
            if self.db.add_sentence(record):
                count += 1
        return count
