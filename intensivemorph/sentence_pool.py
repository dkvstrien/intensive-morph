"""
Soak sentence pool — scores and selects sentences for maximum target-word density.

Unlike AnkiMorphs' T+1 (exactly 1 unknown), Soak seeks high-density sentences:
  - Multiple target words per sentence for efficient soaking
  - Prioritizes words in burst phase
  - Penalizes sentences with mature words (waste of space)
  - Penalizes sentences with unknown non-target words (adds confusion)

Supports two modes:
  - "review" (default): score-based selection for high-density soak sessions
  - "reader": sequential delivery by source/position for immersive reading
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from .config import IntensiveMorphConfig
from .models import LemmaState, SentenceRecord, IntensiveMorphDB, Stage


class SentencePool:
    """
    Manages the pool of available sentences and selects the best
    ones for the daily soak-in session based on target-word density
    and word priority.
    """

    def __init__(self, config: IntensiveMorphConfig, db: IntensiveMorphDB):
        self.config = config
        self.db = db

    def score_sentence(
        self,
        sentence: SentenceRecord,
        lemma_states: Dict[str, LemmaState],
        target_set: Set[str],
    ) -> float:
        """
        Score a sentence for the soak-in session.
        Higher score = more likely to be selected.

        Scoring formula:
          base = sum(priority of each target word in sentence)
          srs_penalty = count_of_SRS_words * srs_penalty
          unknown_penalty = count_of_unknown_non_target * max_unknown_penalty
          mature_penalty = count_of_mature_words * mature_word_penalty

          score = base² - srs_penalty - unknown_penalty - mature_penalty

        The square on base makes high-density sentences explode in score.
        """
        target_words_in_sent: List[LemmaState] = []
        srs_words = 0
        unknown_non_target = 0
        mature_words = 0

        for lemma in sentence.lemmas:
            state = lemma_states.get(lemma)
            if state is None:
                # Unknown lemma — not in our DB
                if lemma in target_set:
                    # It's a target but not tracked yet — create it
                    state = LemmaState(lemma=lemma, is_target=True)
                    self.db.upsert_lemma(state)
                    lemma_states[lemma] = state
                    target_words_in_sent.append(state)
                else:
                    unknown_non_target += 1
                continue

            if state.is_target and state.stage in (Stage.NEW, Stage.SOAK_IN, Stage.BURST):
                target_words_in_sent.append(state)
            elif state.stage == Stage.SRS:
                srs_words += 1
            elif state.stage == Stage.MATURE:
                mature_words += 1
            elif state.is_target and state.stage == Stage.NEW:
                # New target that hasn't started soak yet
                target_words_in_sent.append(state)
            else:
                # Non-target word in any stage — neutral
                pass

        if not target_words_in_sent:
            return 0.0  # No target words, no value

        # Base: sum of priorities squared (explodes for multi-target)
        priority_sum = sum(w.priority for w in target_words_in_sent)
        base_score = priority_sum ** 2

        # Penalties
        srs_penalty = srs_words * self.config.srs_penalty
        unknown_penalty = unknown_non_target * self.config.max_unknown_penalty
        mature_penalty = mature_words * self.config.mature_word_penalty

        score = base_score - srs_penalty - unknown_penalty - mature_penalty
        return max(0.0, score)

    def select_sentences(
        self,
        count: int,
        exclude_sentence_ids: Optional[Set[int]] = None,
        mode: Optional[str] = None,
    ) -> List[Tuple[SentenceRecord, float]]:
        """
        Select the top-N sentences for today's session.

        Args:
            count: Maximum number of sentences to return.
            exclude_sentence_ids: Optional set of sentence IDs to skip.
            mode: "review" (scored by density, default) or "reader" (sequential by source).

        Returns:
            List of (SentenceRecord, score) sorted appropriately for the mode.
        """
        mode = mode or self.config.mode

        if mode == "reader":
            return self._select_reader_sentences(count)

        # Default: review mode — score-based selection
        return self._select_review_sentences(count, exclude_sentence_ids)

    def _select_review_sentences(
        self,
        count: int,
        exclude_sentence_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[SentenceRecord, float]]:
        """Score-based selection for review mode."""
        config = self.config

        # Gather all lemma states
        all_lemmas = self.db.get_all_lemmas()
        lemma_states: Dict[str, LemmaState] = {l.lemma: l for l in all_lemmas}
        target_set = {l.lemma for l in all_lemmas if l.is_target}

        if not target_set:
            return []

        # Score all sentences that contain at least one target lemma
        scored: List[Tuple[SentenceRecord, float]] = []

        for sentence in self.db.get_all_sentences():
            # Check if it has any known target lemmas
            has_target = any(l in target_set for l in sentence.lemmas)
            if not has_target:
                continue

            score = self.score_sentence(sentence, lemma_states, target_set)
            if score > 0:
                scored.append((sentence, score))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:count]

    def _select_reader_sentences(
        self,
        count: int,
    ) -> List[Tuple[SentenceRecord, float]]:
        """
        Sequential selection for reader mode.

        Returns sentences from the active source in position order,
        starting from the current reader progress position.

        Each sentence gets a fake score = its position, so the
        contract (SentenceRecord, float) is preserved.
        """
        source = self.config.reader_active_source
        if not source:
            return []

        sentences = self.db.get_sentences_by_source(source)
        if not sentences:
            return []

        start_pos = self.db.get_reader_progress(source)

        # Find the sentence at or after the start position
        start_idx = 0
        for i, s in enumerate(sentences):
            if s.position >= start_pos:
                start_idx = i
                break

        # Take the next `count` sentences
        batch = sentences[start_idx:start_idx + count]
        return [(s, float(s.position)) for s in batch]

    def select_sentences_for_lemma(
        self, lemma: str, count: int = 5
    ) -> List[Tuple[SentenceRecord, float]]:
        """
        Select the best sentences containing a specific lemma.
        Useful for 1T confirmation tests or AI agents injecting targeted practice.
        """
        all_lemmas = self.db.get_all_lemmas()
        lemma_states: Dict[str, LemmaState] = {l.lemma: l for l in all_lemmas}
        target_set = {l.lemma for l in all_lemmas if l.is_target}

        sentences = self.db.get_sentences_for_lemma(lemma)
        scored = []
        for sentence in sentences:
            score = self.score_sentence(sentence, lemma_states, target_set)
            scored.append((sentence, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:count]

    def compute_density_stats(self) -> dict:
        """
        Compute statistics about sentence density for reporting.
        """
        all_lemmas = self.db.get_all_lemmas()
        target_set = {l.lemma for l in all_lemmas if l.is_target}

        if not target_set or not self.db.get_all_sentences():
            return {"total_sentences": 0, "sentences_with_targets": 0, "avg_density": 0.0}

        total = 0
        with_targets = 0
        densities = []

        for sentence in self.db.get_all_sentences():
            total += 1
            targets = sentence.count_target_lemmas(target_set)
            if targets > 0:
                with_targets += 1
                density = sentence.compute_density(target_set)
                densities.append(density)

        avg_density = sum(densities) / len(densities) if densities else 0.0
        max_density = max(densities) if densities else 0.0

        return {
            "total_sentences": total,
            "sentences_with_targets": with_targets,
            "avg_density": round(avg_density, 3),
            "max_density": round(max_density, 3),
            "pct_with_targets": round(with_targets / total * 100, 1) if total > 0 else 0,
        }
