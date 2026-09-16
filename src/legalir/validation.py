from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from .text import normalize_question


def grouped_folds(questions: list[dict[str, Any]], n_folds: int) -> dict[str, int]:
    """Assign duplicate-normalized questions to the same deterministic fold."""
    assignments: dict[str, int] = {}
    for question in questions:
        group = normalize_question(question["question"])
        digest = hashlib.sha256(group.encode("utf-8")).hexdigest()
        assignments[question["qid"]] = int(digest[:12], 16) % n_folds
    return assignments


def score_predictions(predictions: dict[str, list[str]], questions: list[dict[str, Any]]) -> dict[str, float]:
    recalls: list[float] = []
    precisions: list[float] = []
    for item in questions:
        truth = set(item["answers"])
        prediction = list(predictions.get(item["qid"], []))
        if not truth or not prediction or len(prediction) > 5:
            recalls.append(0.0)
            precisions.append(0.0)
            continue
        hits = len(truth.intersection(prediction))
        recalls.append(hits / len(truth))
        precisions.append(hits / len(prediction))
    return {
        "recall": sum(recalls) / max(1, len(recalls)),
        "precision": sum(precisions) / max(1, len(precisions)),
    }


def score_candidates(predictions: dict[str, list[str]], questions: list[dict[str, Any]]) -> dict[str, float]:
    """Measure a candidate set without applying the official five-result limit."""
    def recall_at(depth: int) -> float:
        values: list[float] = []
        for item in questions:
            truth = set(item["answers"])
            prediction = set(predictions.get(item["qid"], [])[:depth])
            values.append(len(truth.intersection(prediction)) / len(truth) if truth else 0.0)
        return sum(values) / max(1, len(values))

    return {
        "candidate_recall": recall_at(10_000_000),
        "recall_at_5": recall_at(5),
        "recall_at_20": recall_at(20),
        "recall_at_50": recall_at(50),
    }


def oracle_recall(candidates: dict[str, list[str]], questions: list[dict[str, Any]]) -> float:
    values = []
    for item in questions:
        truth = set(item["answers"])
        values.append(len(truth.intersection(candidates.get(item["qid"], []))) / len(truth) if truth else 0.0)
    return sum(values) / max(1, len(values))


def validate_submission_shape(
    submission: dict[str, Any],
    questions: list[dict[str, Any]],
    corpus_ids: set[str],
) -> None:
    expected = {item["qid"] for item in questions}
    if set(submission) != expected:
        missing = expected.difference(submission)
        extra = set(submission).difference(expected)
        raise ValueError(f"Submission question IDs differ (missing={len(missing)}, extra={len(extra)})")
    for qid, value in submission.items():
        answers = value.get("answer") if isinstance(value, dict) else None
        if not isinstance(answers, list) or len(answers) != 5 or len(set(answers)) != 5:
            raise ValueError(f"{qid}: answer must contain exactly five unique document IDs")
        invalid = set(map(str, answers)).difference(corpus_ids)
        if invalid:
            raise ValueError(f"{qid}: unknown document IDs: {sorted(invalid)[:3]}")
