"""
Soak scheduler — nightly maintenance and daily session builder.

Runs as a cron/anacron-like task:
  1. Score and promote lemmas based on Bayesian inference
  2. Select top sentences for today's soak session
  3. Generate 1T confirmation cards for graduating lemmas
  4. Reposition cards in Anki queue (Anki addon mode)
  5. Clean up stale data
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

from .config import IntensiveMorphConfig
from .inference import BayesianInference
from .models import LemmaState, SentenceRecord, IntensiveMorphDB, Stage
from .sentence_pool import SentencePool


class IntensiveMorphScheduler:
    """
    Runs nightly maintenance and builds daily sessions.
    Pure-python core — Anki-specific card operations are in anki_addon/.
    """

    def __init__(self, config: IntensiveMorphConfig, db: IntensiveMorphDB):
        self.config = config
        self.db = db
        self.inference = BayesianInference(config, db)
        self.pool = SentencePool(config, db)

    def process_reviews(
        self,
        reviews: List[dict],
    ) -> dict:
        """
        Process a batch of reviews from any source (desktop, AnkiDroid via sync).

        Each review dict:
            sentence_text: str   — the sentence that was reviewed
            ease: int            — 1=Again, 2=Hard, 3=Good, 4=Easy
            timestamp: float     — unix timestamp of the review

        Returns summary of what was processed.
        """
        target_set = {l.lemma for l in self.db.get_target_lemmas()}
        if not target_set:
            return {"processed": 0, "message": "No target words defined"}

        processed = 0
        for review in reviews:
            # Reconstruct sentence lemmas
            lemmas = self.db.get_all_sentences()
            sentence_lemmas = None
            sentence_id = None
            for s in lemmas:
                if s.text == review["sentence_text"]:
                    sentence_lemmas = s.lemmas
                    break

            if not sentence_lemmas:
                # Sentence not in DB — try to lemmatize on the fly
                # (The Anki addon should have stored lemmas in the note fields)
                continue

            passed = review.get("ease", 3) >= 3
            self.inference.process_review(
                sentence_lemmas,
                passed=passed,
                target_set=target_set,
                sentence_id=sentence_id,
            )
            processed += 1

        return {
            "processed": processed,
            "target_set_size": len(target_set),
        }

    def run_nightly(self) -> dict:
        """
        Full nightly maintenance cycle.
        Returns a report of what was done.

        Steps:
        1. Process overdue SRS lemmas
        2. Check for graduates (soak-in → SRS)
        3. Select top sentences for tomorrow's soak session
        4. Generate 1T confirmations for lemmas near graduation
        5. Clean up stale data
        """
        report: dict = {
            "srs_overdue": 0,
            "graduates_to_srs": 0,
            "graduates_to_mature": 0,
            "sentences_selected": 0,
            "1t_confirmations": 0,
            "stale_removed": 0,
        }

        # 1. Process SRS lemmas
        srs_lemmas = self.db.get_lemmas_by_stage(Stage.SRS)
        now = time.time()
        for lemma in srs_lemmas:
            if lemma.next_due > 0 and now >= lemma.next_due:
                report["srs_overdue"] += 1

            # Check SRS → MATURE graduation
            if lemma.srs_interval >= self.config.srs_max_interval_days:
                lemma.stage = Stage.MATURE
                self.db.upsert_lemma(lemma)
                report["graduates_to_mature"] += 1

        # 2. Check soak-in → SRS graduation
        soaking_lemmas = (
            self.db.get_lemmas_by_stage(Stage.SOAK_IN) +
            self.db.get_lemmas_by_stage(Stage.BURST)
        )
        for lemma in soaking_lemmas:
            old_stage = lemma.stage
            self.inference._check_graduation(lemma)
            if lemma.stage != old_stage:
                if lemma.stage == Stage.SRS:
                    report["graduates_to_srs"] += 1
                self.db.upsert_lemma(lemma)

        # 3. Select sentences for soak session
        selected = self.pool.select_sentences(self.config.daily_soak_quota)
        report["sentences_selected"] = len(selected)

        # 4. Generate 1T confirmations for lemmas near graduation
        threshold = self.config.soak_in_graduate_score * 0.8  # 80% of graduation threshold
        for lemma in soaking_lemmas:
            if (lemma.score >= threshold
                    and lemma.appearances >= self.config.soak_in_min_appearances):
                # This lemma is ready for a 1T confirmation test
                # The actual card generation is Anki-specific, done in anki_addon/
                report["1t_confirmations"] += 1

        # 5. Clean up — mark old NEW lemmas that have no sentences as stale
        new_lemmas = self.db.get_lemmas_by_stage(Stage.NEW)
        for lemma in new_lemmas:
            if self.db.count_sentences_for_lemma(lemma.lemma) == 0:
                # No sentences available — check if it's been NEW for > 7 days
                if lemma.created_at > 0 and (now - lemma.created_at) > 7 * 86400:
                    self.db.delete_lemma(lemma.lemma)
                    report["stale_removed"] += 1

        return report

    def build_daily_session(self) -> dict:
        """
        Build today's study session plan.

        Returns a dict with:
          - soak_cards: list of (SentenceRecord, score) for today
          - srs_cards: list of (LemmaState, SentenceRecord) for due SRS reviews
          - confirmations: list of LemmaState needing 1T tests
        """
        result: dict = {
            "soak_cards": [],
            "srs_cards": [],
            "confirmations": [],
        }

        # --- Soak-in cards ---
        selected = self.pool.select_sentences(self.config.daily_soak_quota)
        result["soak_cards"] = selected

        # --- SRS cards due today ---
        all_lemmas = self.db.get_all_lemmas()
        lemma_states: Dict[str, LemmaState] = {l.lemma: l for l in all_lemmas}
        today = time.time()

        for lemma in all_lemmas:
            if lemma.stage != Stage.SRS:
                continue
            if lemma.next_due > 0 and today < lemma.next_due:
                continue  # Not yet due

            # Find a fresh sentence for this lemma
            sentences = self.db.get_sentences_for_lemma(lemma.lemma)
            # Pick one that hasn't been used recently (or any)
            if sentences:
                result["srs_cards"].append((lemma, sentences[0]))
                if len(result["srs_cards"]) >= self.config.daily_srs_quota:
                    break

        # --- 1T confirmations ---
        for lemma in all_lemmas:
            if lemma.stage not in (Stage.SOAK_IN, Stage.BURST):
                continue
            if (lemma.score >= self.config.soak_in_graduate_score * 0.8
                    and lemma.appearances >= self.config.soak_in_min_appearances):
                result["confirmations"].append(lemma)

        return result

    def apply_srs_review(self, lemma: str, passed: bool) -> dict:
        """
        Process an SRS review for a single lemma.
        Returns updated state.
        """
        state = self.db.get_lemma(lemma)
        if state is None or state.stage != Stage.SRS:
            return {"error": f"Lemma '{lemma}' is not in SRS stage"}

        if passed:
            # SM-2 style interval expansion
            if state.srs_interval == 0:
                state.srs_interval = 1
            elif state.srs_interval == 1:
                state.srs_interval = 3
            else:
                state.srs_interval = min(
                    int(state.srs_interval * 2.5),
                    self.config.srs_max_interval_days
                )
            state.score = min(5.0, state.score + 0.5)
        else:
            # Failed — reset interval
            state.srs_interval = 1
            state.score = max(-5.0, state.score - 0.5)

        state.next_due = time.time() + state.srs_interval * 86400
        state.appearances += 1
        if passed:
            state.pass_count += 1
        else:
            state.fail_count += 1
        state.updated_at = time.time()

        self.db.upsert_lemma(state)

        # Check mature graduation
        if state.srs_interval >= self.config.srs_max_interval_days:
            state.stage = Stage.MATURE
            self.db.upsert_lemma(state)

        return state.to_dict()
