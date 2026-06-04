"""
Soak morphemizer — extracts lemmas from text using language-specific analyzers.

Uses pymorphy3 for Russian by default. Extensible to other languages.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .config import IntensiveMorphConfig


class Morphemizer:
    """
    Extracts lemmas (base forms) from text sentences.

    Uses pymorphy3 for Russian. For other languages, falls back to
    simple lowercasing + stripping punctuation (you'll get surface forms,
    not true lemmas — swap in a proper analyzer per language).
    """

    def __init__(self, config: IntensiveMorphConfig):
        self.config = config
        self._morph = None  # lazy-loaded pymorphy3 instance
        self._stop_words: Set[str] = set()
        self._load_stop_words()

    def _load_stop_words(self) -> None:
        """Load language-appropriate stop words."""
        # Russian stop words (high-frequency function words to ignore)
        self._stop_words = {
            "и", "в", "во", "не", "что", "он", "на", "я", "с", "со",
            "как", "а", "то", "все", "она", "так", "его", "но", "да",
            "ты", "к", "у", "же", "вы", "за", "бы", "по", "только",
            "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
            "о", "из", "ему", "теперь", "когда", "даже", "вдруг",
            "если", "уже", "или", "быть", "был", "была", "были",
            "чем", "ли", "до", "без", "для", "через", "над", "под",
            "об", "при", "про", "ну", "вот", "это", "этого", "этом",
            "этот", "эта", "эти", "этих", "этим", "который", "которая",
            "которые", "чтобы", "также", "очень", "есть", "стал",
            "стала", "стали", "стало", "сказал", "сказала", "сказали",
            "мочь", "могу", "может", "могут", "мог", "могла", "могли",
            "весь", "всё", "вся", "все", "сам", "сама", "сами",
            "самое", "самый", "другой", "другая", "другие", "других",
            "наш", "наша", "наши", "ваш", "ваша", "ваши", "свой",
            "своя", "свои", "свое", "себя", "себе", "собой",
        }

    @property
    def morph(self):
        if self._morph is None:
            import pymorphy3
            self._morph = pymorphy3.MorphAnalyzer(lang=self.config.language)
        return self._morph

    def lemmatize(self, text: str) -> List[str]:
        """
        Extract lemmas from a sentence.
        Returns a list of content-word lemmas in order of appearance.
        """
        words = self._tokenize(text)
        lemmas: List[str] = []

        for word in words:
            if not word or len(word) <= 1:
                continue

            lemma = self._lemmatize_word(word)
            if lemma and lemma not in self._stop_words:
                lemmas.append(lemma)

        return lemmas

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words, preserving Cyrillic and Latin."""
        # Handle punctuation attached to words
        text = re.sub(r'[«»""''„“‟"]', ' ', text)
        text = re.sub(r'[—–\-]', ' ', text)
        # Split on whitespace and common punctuation
        tokens = re.findall(r'[а-яёА-ЯЁa-zA-Z]+(?:-[а-яёА-ЯЁa-zA-Z]+)*', text)
        return tokens

    def _lemmatize_word(self, word: str) -> Optional[str]:
        """Get the normal form (lemma) of a single word."""
        word_lower = word.lower().strip("'\"-")

        if not word_lower or len(word_lower) <= 1:
            return None

        if word_lower in self._stop_words:
            return word_lower  # Still return it (but filtered out by caller)

        # Use pymorphy3 for Russian
        if self.config.language == "ru":
            try:
                parsed = self.morph.parse(word_lower)[0]
                if parsed.score > 0.3:  # Minimum confidence threshold
                    lemma = parsed.normal_form
                    return lemma.lower() if lemma else word_lower
                return word_lower  # Fallback: use surface form
            except Exception:
                return word_lower
        else:
            # For non-Russian: just lowercase and strip
            return word_lower

    def lemmatize_sentences(self, sentences: List[str]) -> List[List[str]]:
        """Batch-lemmatize a list of sentences."""
        return [self.lemmatize(s) for s in sentences]

    def compute_lemma_set(self, sentences: List[str]) -> Dict[str, int]:
        """
        Extract all unique lemmas across sentences with their frequency.
        Returns dict of lemma → count.
        """
        freq: Dict[str, int] = {}
        for text in sentences:
            lemmas = self.lemmatize(text)
            for lemma in lemmas:
                freq[lemma] = freq.get(lemma, 0) + 1
        return freq
