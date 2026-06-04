"""
Soak configuration — user-settings model.

All settings are JSON-serializable and can be changed at runtime.
The Anki addon stores this in config.json; the CLI uses it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntensiveMorphConfig:
    """All configurable settings for Soak."""

    # --- Target language ---
    language: str = "ru"  # ISO code, affects morphemizer choice

    # --- Soak-in phase ---
    soak_in_burst_size: int = 8  # front-loaded appearances in burst phase
    soak_in_burst_hours: int = 48  # burst duration in hours
    soak_in_min_appearances: int = 10  # minimum appearances before graduation
    soak_in_graduate_score: float = 2.5  # Bayesian score threshold for graduation
    soak_in_min_pass_rate: float = 0.7  # minimum pass rate to graduate

    # --- Sentence selection ---
    min_target_density: float = 0.3  # minimum fraction of target words in a sentence
    max_unknown_penalty: float = 5.0  # penalty per unknown non-target word
    srs_penalty: float = 0.5  # penalty per SRS-stage word in sentence
    mature_word_penalty: float = 0.1  # tiny penalty for mature words (noise)

    # --- SRS ---
    srs_max_interval_days: int = 180  # mature threshold
    srs_initial_interval: int = 1  # days after first graduation

    # --- Daily limits ---
    daily_soak_quota: int = 30  # max soak-in cards per day
    daily_srs_quota: int = 15  # max SRS review cards per day

    # --- Confidence scoring ---
    pass_boost_initial: float = 0.15  # initial boost per PASS on multi-word
    pass_boost_min: float = 0.05  # minimum boost per PASS as confidence grows
    fail_penalty_fraction: float = 0.25  # weakest fraction of words that take FAIL penalty
    fail_penalty_initial: float = 0.2  # penalty per FAIL on weakest words
    fail_penalty_min: float = 0.02  # minimum FAIL penalty (for uncertain words)

    # --- Target list ---
    target_list_path: Optional[str] = None  # path to custom target word list file

    # --- Corpus paths ---
    bundled_corpus_path: Optional[str] = None  # path to bundled corpus JSON

    # --- Reader mode ---
    mode: str = "review"  # "review" (scored by density) or "reader" (sequential by source)
    reader_batch_size: int = 1  # sentences per batch in reader mode
    reader_active_source: str = ""  # currently active source in reader mode

    # --- Anki deck settings ---
    deck_name: str = "IntensiveMorph"
    target_tag: str = "soak::target"
    soak_in_tag: str = "soak::soak_in"
    burst_tag: str = "soak::burst"
    srs_tag: str = "soak::srs"
    mature_tag: str = "soak::mature"

    # --- Field names on notes ---
    field_lemma: str = "Soak-Lemma"
    field_stage: str = "Soak-Stage"
    field_score: str = "Soak-Score"
    field_appearances: str = "Soak-Appearances"
    field_burst_count: str = "Soak-BurstCount"
    field_pass_rate: str = "Soak-PassRate"
    field_next_due: str = "Soak-NextDue"
    field_interval: str = "Soak-Interval"

    @classmethod
    def default(cls) -> IntensiveMorphConfig:
        return cls()

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> IntensiveMorphConfig:
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
