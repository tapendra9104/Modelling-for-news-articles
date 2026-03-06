from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")

CUSTOM_STOPWORDS = {
    "also",
    "article",
    "bbc",
    "breaking",
    "cnn",
    "news",
    "reported",
    "report",
    "reuters",
    "said",
    "says",
    "story",
    "today",
}


class TextPreprocessor:
    def __init__(
        self,
        use_spacy: bool = False,
        use_nltk: bool = False,
        extra_stopwords: Iterable[str] | None = None,
    ) -> None:
        self.stop_words = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS | set(extra_stopwords or [])
        self._spacy_model = None
        self._wordnet_lemmatizer = None

        if use_spacy:
            self._spacy_model = self._load_spacy_model()
        if use_nltk and self._spacy_model is None:
            self._wordnet_lemmatizer = self._load_wordnet_lemmatizer()

    def _load_spacy_model(self) -> object | None:
        try:
            import spacy

            return spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            return None

    def _load_wordnet_lemmatizer(self) -> object | None:
        try:
            from nltk.stem import WordNetLemmatizer

            return WordNetLemmatizer()
        except Exception:
            return None

    def clean_text(self, text: str) -> str:
        normalized = URL_PATTERN.sub(" ", text or "")
        normalized = NON_ALPHA_PATTERN.sub(" ", normalized)
        normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip().lower()
        return normalized

    def _lemmatize_token(self, token: str) -> str:
        if self._wordnet_lemmatizer is not None:
            try:
                return str(self._wordnet_lemmatizer.lemmatize(token))
            except LookupError:
                pass

        if token.endswith("ies") and len(token) > 4:
            return f"{token[:-3]}y"
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 4:
            return token[:-1]
        return token

    def tokenize(self, text: str) -> list[str]:
        if self._spacy_model is not None:
            doc = self._spacy_model(text)
            return [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and len(token.text) > 2 and token.lemma_.lower() not in self.stop_words
            ]

        tokens: list[str] = []
        for token in TOKEN_PATTERN.findall(text):
            candidate = self._lemmatize_token(token.lower())
            if candidate not in self.stop_words and len(candidate) > 2:
                tokens.append(candidate)
        return tokens

    def preprocess_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        processed = frame.copy()
        processed["combined_text"] = (
            processed["title"].fillna("").astype(str) + " " + processed["content"].fillna("").astype(str)
        ).str.strip()
        processed["clean_text"] = processed["combined_text"].map(self.clean_text)
        processed["tokens"] = processed["clean_text"].map(self.tokenize)
        processed["processed_text"] = processed["tokens"].map(" ".join)
        processed = processed[processed["processed_text"].str.len() > 0].reset_index(drop=True)
        return processed
