"""
Soak data models — lemma states, sentence records, and DB schema.

This module defines the core data structures that drive soak mode:
- LemmaState: tracks a single word through soak-in → SRS → mature
- SentenceRecord: a sentence with its lemma composition
- IntensiveMorphDB: SQLite database for persistence
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class Stage(Enum):
    NEW = "new"
    SOAK_IN = "soak_in"
    BURST = "burst"  # first 48h of soak_in
    SRS = "srs"
    MATURE = "mature"


@dataclass
class LemmaState:
    """Tracks a single lemma through the soak pipeline."""

    lemma: str
    stage: Stage = Stage.NEW
    score: float = 0.0  # Bayesian confidence (-5 to +5)
    appearances: int = 0  # total times seen in sentences
    burst_count: int = 0  # appearances in current burst window
    burst_started_at: float = 0.0  # unix timestamp of first burst appearance
    pass_count: int = 0
    fail_count: int = 0
    srs_interval: int = 0  # current SRS interval in days
    next_due: float = 0.0  # unix timestamp of next SRS review
    is_target: bool = False  # user-marked as target word
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def pass_rate(self) -> float:
        total = self.pass_count + self.fail_count
        if total == 0:
            return 0.0
        return self.pass_count / total

    @property
    def is_bursting(self) -> bool:
        """Returns True if still in the 48h burst window."""
        if self.burst_started_at == 0:
            return False
        return (time.time() - self.burst_started_at) < 48 * 3600

    @property
    def priority(self) -> float:
        """
        Priority score for sentence selection.
        Higher = more important to show this word.
        """
        if self.stage == Stage.MATURE:
            return 0.0
        if self.stage == Stage.SRS:
            if self.next_due > 0 and time.time() >= self.next_due:
                return 5.0  # due SRS review is high priority
            return 1.0  # not yet due
        # Soak-in / Burst
        if self.is_bursting:
            burst_remaining = max(0, 8 - self.burst_count)
            return burst_remaining  # 8,7,6...1 — high urgency
        # Post-burst decay
        days_in = (time.time() - self.burst_started_at) / 86400 if self.burst_started_at > 0 else 0
        base = 1.0 / (1.0 + days_in) ** 2
        return base * 5  # scale up to match burst range

    def to_dict(self) -> dict:
        return {
            "lemma": self.lemma,
            "stage": self.stage.value,
            "score": self.score,
            "appearances": self.appearances,
            "burst_count": self.burst_count,
            "burst_started_at": self.burst_started_at,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "srs_interval": self.srs_interval,
            "next_due": self.next_due,
            "is_target": self.is_target,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LemmaState:
        d = dict(d)
        d["stage"] = Stage(d["stage"])
        return cls(**d)


@dataclass
class SentenceRecord:
    """A sentence with its lemma composition and metadata."""

    text: str  # original sentence text
    translation: str = ""  # optional translation
    lemmas: List[str] = field(default_factory=list)  # all lemmas in the sentence
    source: str = ""  # e.g., filename, corpus name
    is_user_sentence: bool = False  # user-injected via API

    def count_target_lemmas(self, target_set: Set[str]) -> int:
        """Count how many lemmas in this sentence are in the target set."""
        return sum(1 for l in self.lemmas if l in target_set)

    def compute_density(self, target_set: Set[str]) -> float:
        """
        Fraction of lemmas in the sentence that are target words.
        Returns 0.0 if no lemmas, or if no lemmas are targets.
        """
        if not self.lemmas:
            return 0.0
        return self.count_target_lemmas(target_set) / len(self.lemmas)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "translation": self.translation,
            "lemmas": self.lemmas,
            "source": self.source,
            "is_user_sentence": self.is_user_sentence,
        }


class IntensiveMorphDB:
    """
    SQLite-backed persistence for soak state.
    Stored separately from Anki's collection.anki2.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with self.con:
            self.con.executescript("""
                CREATE TABLE IF NOT EXISTS lemmas (
                    lemma TEXT PRIMARY KEY,
                    stage TEXT NOT NULL DEFAULT 'new',
                    score REAL NOT NULL DEFAULT 0.0,
                    appearances INTEGER NOT NULL DEFAULT 0,
                    burst_count INTEGER NOT NULL DEFAULT 0,
                    burst_started_at REAL NOT NULL DEFAULT 0.0,
                    pass_count INTEGER NOT NULL DEFAULT 0,
                    fail_count INTEGER NOT NULL DEFAULT 0,
                    srs_interval INTEGER NOT NULL DEFAULT 0,
                    next_due REAL NOT NULL DEFAULT 0.0,
                    is_target INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL DEFAULT (unixepoch()),
                    updated_at REAL NOT NULL DEFAULT (unixepoch())
                );

                CREATE TABLE IF NOT EXISTS sentences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL UNIQUE,
                    translation TEXT NOT NULL DEFAULT '',
                    lemmas TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT '',
                    is_user_sentence INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL DEFAULT (unixepoch())
                );

                CREATE TABLE IF NOT EXISTS sentence_lemma_map (
                    sentence_id INTEGER NOT NULL,
                    lemma TEXT NOT NULL,
                    FOREIGN KEY (sentence_id) REFERENCES sentences(id),
                    FOREIGN KEY (lemma) REFERENCES lemmas(lemma),
                    PRIMARY KEY (sentence_id, lemma)
                );

                CREATE TABLE IF NOT EXISTS review_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lemma TEXT NOT NULL,
                    sentence_id INTEGER,
                    passed INTEGER NOT NULL,
                    score_before REAL NOT NULL,
                    score_after REAL NOT NULL,
                    reviewed_at REAL NOT NULL DEFAULT (unixepoch()),
                    FOREIGN KEY (lemma) REFERENCES lemmas(lemma),
                    FOREIGN KEY (sentence_id) REFERENCES sentences(id)
                );

                CREATE INDEX IF NOT EXISTS idx_lemmas_stage ON lemmas(stage);
                CREATE INDEX IF NOT EXISTS idx_lemmas_is_target ON lemmas(is_target);
                CREATE INDEX IF NOT EXISTS idx_sentence_lemma ON sentence_lemma_map(lemma);
                CREATE INDEX IF NOT EXISTS idx_review_log_lemma ON review_log(lemma);
            """)

    # --- Lemma CRUD ---

    def upsert_lemma(self, state: LemmaState) -> None:
        now = time.time()
        with self.con:
            self.con.execute("""
                INSERT INTO lemmas (lemma, stage, score, appearances, burst_count,
                    burst_started_at, pass_count, fail_count, srs_interval, next_due,
                    is_target, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, unixepoch()), ?)
                ON CONFLICT(lemma) DO UPDATE SET
                    stage = excluded.stage,
                    score = excluded.score,
                    appearances = excluded.appearances,
                    burst_count = excluded.burst_count,
                    burst_started_at = excluded.burst_started_at,
                    pass_count = excluded.pass_count,
                    fail_count = excluded.fail_count,
                    srs_interval = excluded.srs_interval,
                    next_due = excluded.next_due,
                    is_target = excluded.is_target,
                    updated_at = excluded.updated_at
            """, (
                state.lemma, state.stage.value, state.score, state.appearances,
                state.burst_count, state.burst_started_at, state.pass_count,
                state.fail_count, state.srs_interval, state.next_due,
                1 if state.is_target else 0,
                state.created_at if state.created_at > 0 else None,
                now,
            ))

    def get_lemma(self, lemma: str) -> Optional[LemmaState]:
        row = self.con.execute(
            "SELECT * FROM lemmas WHERE lemma = ?", (lemma,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_lemma(row)

    def get_all_lemmas(self) -> List[LemmaState]:
        rows = self.con.execute("SELECT * FROM lemmas ORDER BY lemma").fetchall()
        return [self._row_to_lemma(r) for r in rows]

    def get_target_lemmas(self) -> List[LemmaState]:
        rows = self.con.execute(
            "SELECT * FROM lemmas WHERE is_target = 1 ORDER BY lemma"
        ).fetchall()
        return [self._row_to_lemma(r) for r in rows]

    def get_lemmas_by_stage(self, stage: Stage) -> List[LemmaState]:
        rows = self.con.execute(
            "SELECT * FROM lemmas WHERE stage = ? ORDER BY lemma", (stage.value,)
        ).fetchall()
        return [self._row_to_lemma(r) for r in rows]

    def delete_lemma(self, lemma: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM lemmas WHERE lemma = ?", (lemma,))
            self.con.execute("DELETE FROM review_log WHERE lemma = ?", (lemma,))

    @staticmethod
    def _row_to_lemma(row: sqlite3.Row) -> LemmaState:
        return LemmaState(
            lemma=row["lemma"],
            stage=Stage(row["stage"]),
            score=row["score"],
            appearances=row["appearances"],
            burst_count=row["burst_count"],
            burst_started_at=row["burst_started_at"],
            pass_count=row["pass_count"],
            fail_count=row["fail_count"],
            srs_interval=row["srs_interval"],
            next_due=row["next_due"],
            is_target=bool(row["is_target"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # --- Sentence CRUD ---

    def add_sentence(self, record: SentenceRecord) -> int:
        with self.con:
            cursor = self.con.execute("""
                INSERT OR IGNORE INTO sentences
                    (text, translation, lemmas, source, is_user_sentence)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record.text, record.translation,
                json.dumps(record.lemmas),
                record.source,
                1 if record.is_user_sentence else 0,
            ))
            sentence_id = cursor.lastrowid

            if sentence_id and record.lemmas:
                # Insert into sentence_lemma_map
                self.con.executemany(
                    "INSERT OR IGNORE INTO sentence_lemma_map (sentence_id, lemma) VALUES (?, ?)",
                    [(sentence_id, l) for l in record.lemmas]
                )
            return sentence_id or 0

    def get_sentence(self, sentence_id: int) -> Optional[SentenceRecord]:
        row = self.con.execute(
            "SELECT * FROM sentences WHERE id = ?", (sentence_id,)
        ).fetchone()
        if row is None:
            return None
        return SentenceRecord(
            text=row["text"],
            translation=row["translation"],
            lemmas=json.loads(row["lemmas"]),
            source=row["source"],
            is_user_sentence=bool(row["is_user_sentence"]),
        )

    def get_all_sentences(self) -> List[SentenceRecord]:
        rows = self.con.execute(
            "SELECT * FROM sentences ORDER BY id"
        ).fetchall()
        return [
            SentenceRecord(
                text=r["text"],
                translation=r["translation"],
                lemmas=json.loads(r["lemmas"]),
                source=r["source"],
                is_user_sentence=bool(r["is_user_sentence"]),
            )
            for r in rows
        ]

    def get_sentences_for_lemma(self, lemma: str) -> List[SentenceRecord]:
        """Get all sentences containing a specific lemma."""
        rows = self.con.execute("""
            SELECT s.* FROM sentences s
            INNER JOIN sentence_lemma_map slm ON s.id = slm.sentence_id
            WHERE slm.lemma = ?
            ORDER BY s.id
        """, (lemma,)).fetchall()
        return [
            SentenceRecord(
                text=r["text"],
                translation=r["translation"],
                lemmas=json.loads(r["lemmas"]),
                source=r["source"],
                is_user_sentence=bool(r["is_user_sentence"]),
            )
            for r in rows
        ]

    def count_sentences_for_lemma(self, lemma: str) -> int:
        row = self.con.execute("""
            SELECT COUNT(*) as cnt FROM sentence_lemma_map WHERE lemma = ?
        """, (lemma,)).fetchone()
        return row["cnt"] if row else 0

    # --- Review log ---

    def log_review(self, lemma: str, sentence_id: Optional[int],
                   passed: bool, score_before: float, score_after: float) -> None:
        with self.con:
            self.con.execute("""
                INSERT INTO review_log
                    (lemma, sentence_id, passed, score_before, score_after)
                VALUES (?, ?, ?, ?, ?)
            """, (lemma, sentence_id, 1 if passed else 0, score_before, score_after))

    def get_review_history(self, lemma: str, limit: int = 50) -> List[dict]:
        rows = self.con.execute("""
            SELECT * FROM review_log
            WHERE lemma = ?
            ORDER BY reviewed_at DESC
            LIMIT ?
        """, (lemma, limit)).fetchall()
        return [dict(r) for r in rows]

    # --- Target list management ---

    def set_targets(self, lemmas: List[str]) -> int:
        """
        Set the target word list. Marks these lemmas as targets,
        unmarks any existing targets not in the list.
        Returns count of targets set.
        """
        with self.con:
            # Unmark all non-burst targets
            self.con.execute("""
                UPDATE lemmas SET is_target = 0, updated_at = unixepoch()
                WHERE is_target = 1
            """)
            # Insert or update each target
            now = time.time()
            for lemma in lemmas:
                self.con.execute("""
                    INSERT INTO lemmas (lemma, stage, is_target, created_at, updated_at)
                    VALUES (?, 'new', 1, ?, ?)
                    ON CONFLICT(lemma) DO UPDATE SET
                        is_target = 1,
                        updated_at = ?
                """, (lemma, now, now, now))
            return len(lemmas)

    def add_target(self, lemma: str) -> bool:
        """Add a single target word. Returns True if added, False if already target."""
        existing = self.get_lemma(lemma)
        if existing and existing.is_target:
            return False
        now = time.time()
        if existing:
            with self.con:
                self.con.execute("""
                    UPDATE lemmas SET is_target = 1, updated_at = ? WHERE lemma = ?
                """, (now, lemma))
        else:
            state = LemmaState(lemma=lemma, is_target=True, created_at=now, updated_at=now)
            self.upsert_lemma(state)
        return True

    def remove_target(self, lemma: str) -> bool:
        """Remove a target word. Returns True if it was a target."""
        existing = self.get_lemma(lemma)
        if not existing or not existing.is_target:
            return False
        with self.con:
            self.con.execute("""
                UPDATE lemmas SET is_target = 0, updated_at = unixepoch()
                WHERE lemma = ?
            """, (lemma,))
        return True

    def get_target_sentences(self) -> List[Tuple[SentenceRecord, Set[str]]]:
        """
        Get all sentences that contain at least one target lemma,
        paired with the set of target lemmas they contain.
        """
        target_lemmas = {l.lemma for l in self.get_target_lemmas()}
        if not target_lemmas:
            return []
        results = []
        for sent in self.get_all_sentences():
            targets_in_sent = {l for l in sent.lemmas if l in target_lemmas}
            if targets_in_sent:
                results.append((sent, targets_in_sent))
        return results

    # --- Maintenance ---

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> IntensiveMorphDB:
        return self

    def __exit__(self, *args) -> None:
        self.close()
