from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .text import strip_accents


class LexicalIndex:
    """Sparse lexical channels: word BM25 and accentless character n-grams.

    The word matrix is preweighted with the BM25 saturation and length
    normalization terms. It is intentionally model-free and separate from the
    neural-model parameter budget.
    """

    def __init__(self, word: CountVectorizer, char: TfidfVectorizer, word_matrix, char_matrix):
        self.word = word
        self.char = char
        self.word_matrix = word_matrix
        self.char_matrix = char_matrix

    @classmethod
    def build(cls, texts: list[str]) -> "LexicalIndex":
        word = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w/-]+\b",
            dtype=np.float32,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        counts = word.fit_transform(texts).tocsr().astype(np.float32)
        document_lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = max(float(document_lengths.mean()), 1.0)
        k1, b = 1.5, 0.75
        denominator_offset = k1 * (1.0 - b + b * document_lengths / average_length)
        bm25 = counts.copy()
        row_ids = np.repeat(np.arange(bm25.shape[0]), np.diff(bm25.indptr))
        bm25.data = bm25.data * (k1 + 1.0) / (bm25.data + denominator_offset[row_ids])
        document_frequency = np.asarray(counts.getnnz(axis=0)).ravel()
        idf = np.log(1.0 + (len(texts) - document_frequency + 0.5) / (document_frequency + 0.5))
        bm25 = bm25 @ sparse.diags(idf.astype(np.float32))
        return cls(word, char, bm25.tocsr(), char.fit_transform([strip_accents(text) for text in texts]))

    def search(self, query: str, channel: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        if channel == "bm25":
            query_counts = self.word.transform([query]).tocsr()
            query_counts.data[:] = 1.0
            scores = (self.word_matrix @ query_counts.T).toarray().ravel()
        elif channel == "accent_char":
            scores = (self.char_matrix @ self.char.transform([strip_accents(query)]).T).toarray().ravel()
        else:
            raise ValueError(f"Unknown lexical channel: {channel}")
        count = min(k, len(scores))
        indices = np.argpartition(scores, -count)[-count:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return indices, scores[indices]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "LexicalIndex":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
