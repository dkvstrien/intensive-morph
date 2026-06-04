"""
Soak Bayesian inference engine — multi-word sentence rating → per-lemma confidence.

Core idea: When a sentence with multiple target words is rated PASS or FAIL,
we distribute credit/penalty across words based on uncertainty.
Uncertain words learn faster. Confident words are barely affected.

The math uses a simple log-odds model:
  - score = log(p / (1-p)) roughly, ranging -5 to +5
  - PASS: boost all target words, weighted by uncertainty
  - FAIL: penalize the weakest fraction of words
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from .config import IntensiveMorphConfig
from .models import LemmaState, IntensiveMorphDB, Stage


class BayesianInference:
    """
    Updates LemmaState scores based on PASS/FAIL ratings of
    sentences containing multiple target words.
    """

    def __init__(self, config: IntensiveMorphConfig, db: IntensiveMorphDB):
        self.config = config
        self.db = db

    def process_review(
        self,
        sentence_lemmas: List[str],
        passed: bool,
        target_set: Optional[Set[str]] = None,
        sentence_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Process a PASS/FAIL rating on a sentence.

        Args:
            sentence_lemmas: All lemmas in the reviewed sentence.
            passed: True if the user rated PASS (3+), False for FAIL (1-2).
            target_set: Set of lemmas to consider (if None, uses all soak-in + SRS).
            sentence_id: Optional DB sentence ID for logging.

        Returns:
            Dict mapping lemma → score after update.
        """
        # Get the lemmas we're tracking in this sentence
        tracked = self._get_tracked_lemmas(sentence_lemmas, target_set)
        if not tracked:
            return {}

        scores_before = {t.lemma: t.score for t in tracked}

        if passed:
            self._apply_pass(tracked)
        else:
            self._apply_fail(tracked)

        # Persist updates
        for t in tracked:
            t.updated_at = __import__("time").time()
            self.db.upsert_lemma(t)

            # Log the review
            if sentence_id:
                self.db.log_review(
                    lemma=t.lemma,
                    sentence_id=sentence_id,
                    passed=passed,
                    score_before=scores_before[t.lemma],
                    score_after=t.score,
                )

            # Check graduation
            self._check_graduation(t)

        return {t.lemma: t.score for t in tracked}

    def process_1t_review(
        self, lemma: str, passed: bool
    ) -> float:
        """
        Process a 1T (one-target) confirmation test.
        Single lemma, unambiguous signal.
        """
        state = self.db.get_lemma(lemma)
        if state is None:
            state = LemmaState(lemma=lemma)
            self.db.upsert_lemma(state)

        score_before = state.score

        if passed:
            # Stronger adjustment for 1T — no ambiguity
            delta = self._confidence_weighted_delta(state.score, base=0.3)
            state.score = min(5.0, state.score + delta)
            state.pass_count += 1
        else:
            delta = self._confidence_weighted_delta(state.score, base=0.25)
            state.score = max(-5.0, state.score - delta)
            state.fail_count += 1

        state.appearances += 1
        state.updated_at = __import__("time").time()
        self.db.upsert_lemma(state)

        self._check_graduation(state)
        return state.score

    def _get_tracked_lemmas(
        self, sentence_lemmas: List[str],
        target_set: Optional[Set[str]] = None,
    ) -> List[LemmaState]:
        """Get LemmaStates for lemmas we're currently tracking.

        Includes NEW targets (auto-promotes to burst on first encounter),
        and SOAK_IN / BURST / SRS words.
        """
        tracked: List[LemmaState] = []

        # Get all lemmas we know about
        all_known = {l.lemma: l for l in self.db.get_all_lemmas()}

        for lemma_str in sentence_lemmas:
            if target_set and lemma_str not in target_set:
                continue  # Not in our target set

            state = all_known.get(lemma_str)
            if state is None:
                # Unknown word — create automatically as target
                if target_set and lemma_str in target_set:
                    state = LemmaState(lemma=lemma_str, is_target=True)
                    self.db.upsert_lemma(state)
                else:
                    continue

            # Track: NEW targets, soaking, burst, and SRS words
            if state.stage in (Stage.NEW, Stage.SOAK_IN, Stage.BURST, Stage.SRS):
                # Auto-promote NEW targets to BURST on first encounter
                if state.stage == Stage.NEW:
                    state.stage = Stage.BURST
                    state.burst_count = 1
                    state.burst_started_at = __import__("time").time()
                    state.updated_at = __import__("time").time()
                    self.db.upsert_lemma(state)
                tracked.append(state)

        return tracked

    def _apply_pass(self, tracked: List[LemmaState]) -> None:
        """
        PASS: all tracked words get a boost proportional to their uncertainty.
        More uncertain words learn more from a confirmed sentence.
        """
        for state in tracked:
            delta = self._confidence_weighted_delta(state.score, base=self.config.pass_boost_initial)
            state.score = min(5.0, state.score + delta)
            state.appearances += 1
            state.pass_count += 1

            # Update burst tracking
            if state.stage == Stage.BURST:
                state.burst_count += 1
                self._maybe_end_burst(state)
            elif state.stage == Stage.SOAK_IN and state.burst_count == 0:
                state.burst_count += 1
                state.burst_started_at = __import__("time").time()
                state.stage = Stage.BURST

    def _apply_fail(self, tracked: List[LemmaState]) -> None:
        """
        FAIL: penalize only the weakest fraction of words.
        This concentrates learning on what's actually unknown.
        """
        # Sort by score ascending — weakest first
        sorted_words = sorted(tracked, key=lambda s: s.score)

        # Take the weakest fraction
        n_culprits = max(1, int(len(sorted_words) * self.config.fail_penalty_fraction))
        culprits = sorted_words[:n_culprits]

        for state in tracked:
            state.appearances += 1

            if state in culprits:
                delta = self._confidence_weighted_delta(state.score, base=self.config.fail_penalty_initial)
                state.score = max(-5.0, state.score - delta)
                state.fail_count += 1

                # Any FAIL resets burst progress
                if state.stage == Stage.BURST:
                    state.stage = Stage.SOAK_IN
                    state.burst_count = 0
            else:
                # Non-culprit: slight positive signal (they didn't cause the fail)
                tiny_boost = self._confidence_weighted_delta(state.score, base=0.02)
                state.score = min(5.0, state.score + tiny_boost)
                state.pass_count += 1

    def _confidence_weighted_delta(self, score: float, base: float) -> float:
        """
        Higher uncertainty = larger adjustment.
        Score close to 0 = high uncertainty → big delta.
        Score near ±5 = high confidence → tiny delta.

        Uses a Gaussian-like weighting centered at 0.
        """
        # Normalize score to 0-1 range (|score| / 5)
        confidence = abs(score) / 5.0
        # Weight: 1.0 at score=0, ~0.14 at score=±5
        weight = math.exp(-3.0 * confidence ** 2)
        # Ensure minimum adjustment
        min_weight = 0.15  # at max confidence, still 15% of base

        # Scale boost based on how negative the score is (catch-up effect)
        if score < 0:
            # Negative scores get extra boost on PASS
            catchup = 1.0 + min(1.0, abs(score) / 2.5)
            weight *= catchup

        effective_weight = max(min_weight, weight)
        return base * effective_weight

    def _maybe_end_burst(self, state: LemmaState) -> None:
        """Transition from BURST to SOAK_IN once burst quota is met."""
        if state.burst_count >= self.config.soak_in_burst_size:
            state.stage = Stage.SOAK_IN

    def _check_graduation(self, state: LemmaState) -> None:
        """
        Check if a lemma is ready to graduate to the next stage.

        Rules:
        - BURST → SOAK_IN: burst_count >= burst_size (handled in _apply_pass)
        - SOAK_IN → SRS: score >= graduate_score AND appearances >= min_appearances
                          AND pass_rate >= min_pass_rate
        - SRS → MATURE: interval >= srs_max_interval_days
        """
        config = self.config

        if state.stage in (Stage.BURST, Stage.SOAK_IN):
            if (state.score >= config.soak_in_graduate_score
                    and state.appearances >= config.soak_in_min_appearances
                    and state.pass_rate >= config.soak_in_min_pass_rate):
                state.stage = Stage.SRS
                state.srs_interval = config.srs_initial_interval
                state.next_due = __import__("time").time() + config.srs_initial_interval * 86400

        elif state.stage == Stage.SRS:
            if state.srs_interval >= config.srs_max_interval_days:
                state.stage = Stage.MATURE

    def get_confusion_report(self, lemma: str) -> dict:
        """
        Generate a machine-readable confusion report for a lemma.
        Useful for AI agents inspecting the system.
        """
        state = self.db.get_lemma(lemma)
        if state is None:
            return {"lemma": lemma, "error": "not found"}

        sentences = self.db.get_sentences_for_lemma(lemma)
        history = self.db.get_review_history(lemma, limit=20)

        recent_sentences = []
        for sent in sentences[:10]:
            recent_sentences.append({
                "text": sent.text,
                "translation": sent.translation,
                "co_lemmas": [l for l in sent.lemmas if l != lemma],
            })

        return {
            "lemma": lemma,
            "stage": state.stage.value,
            "score": round(state.score, 2),
            "appearances": state.appearances,
            "burst_count": state.burst_count,
            "pass_rate": round(state.pass_rate, 2),
            "inference_confidence": round(1.0 - abs(state.score) / 5.0, 2),
            "is_target": state.is_target,
            "recent_reviews": [
                {
                    "passed": bool(h["passed"]),
                    "score_before": round(h["score_before"], 2),
                    "score_after": round(h["score_after"], 2),
                }
                for h in history[:10]
            ],
            "recent_sentences": recent_sentences,
        }
