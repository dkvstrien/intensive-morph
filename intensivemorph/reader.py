"""
Reader mode — sequential sentence immersion for IntensiveMorph.

Read a book or story one sentence at a time, rate each PASS/FAIL,
and let the Bayesian engine silently build up lemma knowledge.

Key design:
  - User reads in original order (by source, position)
  - Each sentence is rated PASS/FAIL like normal
  - Progress is tracked per source in the DB (reader_progress table)
  - Switch back to review mode anytime for scored density selection
  - The BayesianInference learns which words are weak regardless of mode
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .config import IntensiveMorphConfig
from .models import SentenceRecord, IntensiveMorphDB


class ReaderSession:
    """
    Manages sequential sentence reading with progress tracking.

    Typical usage:
        reader = ReaderSession(config, db)
        reader.start_source("My Book.txt")  # or pick from get_sources()
        batch = reader.next_batch(size=5)    # get next 5 sentences
        # ... rate each sentence PASS/FAIL ...
        reader.advance(5)                   # save progress
        batch = reader.next_batch(size=5)   # continue reading
        reader.prev_batch(size=3)           # go back for review
    """

    def __init__(self, config: IntensiveMorphConfig, db: IntensiveMorphDB):
        self.config = config
        self.db = db
        self._sentences_cache: Dict[str, List[SentenceRecord]] = {}

    def get_sources(self) -> List[dict]:
        """
        List available sources with metadata.

        Returns:
            List of dicts: {source, total_sentences, current_position, progress_pct}
        """
        all_sources = self.db.get_sources()
        result = []
        for source in all_sources:
            sentences = self.__get_cached_sentences(source)
            total = len(sentences)
            if total == 0:
                continue
            current = self.db.get_reader_progress(source)
            pct = round(current / total * 100, 1) if total > 0 else 0.0
            result.append({
                "source": source,
                "total_sentences": total,
                "current_position": min(current, total - 1) if total > 0 else 0,
                "current_sentence": sentences[min(current, total - 1)].text[:80] if total > 0 else "",
                "progress_pct": min(pct, 100.0),
                "remaining": max(0, total - current),
            })
        return result

    def start_source(self, source: str) -> dict:
        """
        Switch to a source and set it as the active reader source.

        Returns a summary of the source.
        """
        sentences = self.db.get_sentences_by_source(source)
        if not sentences:
            return {"error": f"No sentences found for source: {source}"}

        self.config.reader_active_source = source

        # Get current position
        current_pos = self.db.get_reader_progress(source)
        if current_pos >= len(sentences):
            current_pos = max(0, len(sentences) - 1)
            self.db.set_reader_progress(source, current_pos)

        current_sentence = sentences[current_pos]
        return {
            "source": source,
            "total": len(sentences),
            "current_position": current_pos,
            "current_sentence": current_sentence.text[:80],
            "progress_pct": round(current_pos / len(sentences) * 100, 1),
        }

    def next_batch(self, size: Optional[int] = None) -> List[SentenceRecord]:
        """
        Get the next batch of sentences from the active source.

        Args:
            size: Number of sentences to return. Defaults to config.reader_batch_size.

        Returns:
            List of SentenceRecord objects, or empty list if no more.
        """
        source = self.config.reader_active_source
        if not source:
            return []

        sentences = self.__get_cached_sentences(source)
        if not sentences:
            return []

        batch_size = size or self.config.reader_batch_size
        current_pos = self.db.get_reader_progress(source)

        # Clamp to valid range
        if current_pos >= len(sentences):
            return []

        batch = sentences[current_pos:current_pos + batch_size]
        return batch

    def advance(self, count: int = 1) -> dict:
        """
        Advance the reading position by count sentences.

        Returns updated position info.
        """
        source = self.config.reader_active_source
        if not source:
            return {"error": "No active source"}

        sentences = self.__get_cached_sentences(source)
        total = len(sentences)
        current = self.db.get_reader_progress(source)
        new_pos = min(current + count, total)

        self.db.set_reader_progress(source, new_pos)
        self.__clear_cache(source)

        return {
            "source": source,
            "position": new_pos,
            "total": total,
            "progress_pct": round(new_pos / total * 100, 1) if total > 0 else 0.0,
            "remaining": max(0, total - new_pos),
        }

    def prev_batch(self, size: Optional[int] = None) -> List[SentenceRecord]:
        """
        Go back by size sentences.

        Returns the current (repositioned) batch.
        """
        source = self.config.reader_active_source
        if not source:
            return []

        sentences = self.__get_cached_sentences(source)
        if not sentences:
            return []

        batch_size = size or self.config.reader_batch_size
        current_pos = self.db.get_reader_progress(source)
        new_pos = max(0, current_pos - batch_size)

        self.db.set_reader_progress(source, new_pos)
        self.__clear_cache(source)

        batch = sentences[new_pos:new_pos + batch_size]
        return batch

    def jump_to(self, position: int) -> List[SentenceRecord]:
        """
        Jump to a specific position in the active source.

        Returns the batch starting at that position.
        """
        source = self.config.reader_active_source
        if not source:
            return []

        sentences = self.__get_cached_sentences(source)
        if not sentences:
            return []

        pos = max(0, min(position, len(sentences) - 1))
        self.db.set_reader_progress(source, pos)
        self.__clear_cache(source)

        batch_size = self.config.reader_batch_size
        return sentences[pos:pos + batch_size]

    def status(self) -> dict:
        """
        Get current reading status.
        """
        source = self.config.reader_active_source
        if not source:
            return {"error": "No active source"}

        sentences = self.__get_cached_sentences(source)
        if not sentences:
            return {"error": f"No sentences for source: {source}"}

        current_pos = self.db.get_reader_progress(source)
        if current_pos >= len(sentences):
            current_pos = max(0, len(sentences) - 1)

        current_sentence = sentences[current_pos]
        return {
            "source": source,
            "total": len(sentences),
            "current_position": current_pos,
            "current_sentence": current_sentence.text[:80],
            "progress_pct": round(current_pos / len(sentences) * 100, 1),
            "remaining": max(0, len(sentences) - current_pos),
            "lemmas_in_current": len(current_sentence.lemmas),
        }

    def get_sentence_at(self, position: int) -> Optional[SentenceRecord]:
        """
        Get a specific sentence by its position in the active source.
        """
        source = self.config.reader_active_source
        if not source:
            return None

        sentences = self.__get_cached_sentences(source)
        if not sentences or position < 0 or position >= len(sentences):
            return None
        return sentences[position]

    # --- Internal ---

    def __get_cached_sentences(self, source: str) -> List[SentenceRecord]:
        if source not in self._sentences_cache:
            self._sentences_cache[source] = self.db.get_sentences_by_source(source)
        return self._sentences_cache[source]

    def __clear_cache(self, source: Optional[str] = None) -> None:
        if source:
            self._sentences_cache.pop(source, None)
        else:
            self._sentences_cache.clear()
