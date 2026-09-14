from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Iterable

from .validation import score_predictions


def rrf(rankings: dict[str, list[str]], weights: dict[str, float], k: int, limit: int) -> list[str]:
    scores: defaultdict[str, float] = defaultdict(float)
    for channel, documents in rankings.items():
        weight = weights.get(channel, 1.0)
        for rank, document_id in enumerate(documents, start=1):
            scores[str(document_id)] += weight / (k + rank)
    return [document_id for document_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def coordinate_search(
    channel_rankings: dict[str, dict[str, list[str]]],
    questions: list[dict],
    k_values: Iterable[int],
    weight_values: Iterable[float],
    limit: int,
    initial_weights: dict[str, float] | None = None,
) -> dict:
    """Small deterministic coordinate search, ordered by official Recall then Precision."""
    channels = list(channel_rankings)
    weights = initial_weights or {channel: 1.0 for channel in channels}
    best: dict | None = None
    for rrf_k in k_values:
        current = dict(weights)
        improved = True
        # The search space is tiny. A finite number of passes avoids an unlikely
        # coordinate oscillation on tied integer-valued Recall scores.
        for _ in range(4):
            if not improved:
                break
            improved = False
            for channel in channels:
                candidate_best = None
                for value in weight_values:
                    trial = dict(current)
                    trial[channel] = value
                    predictions = {
                        item["qid"]: rrf(
                            {name: ranks[item["qid"]] for name, ranks in channel_rankings.items()},
                            trial,
                            rrf_k,
                            limit,
                        )
                        for item in questions
                    }
                    metrics = score_predictions(predictions, questions)
                    result = {"weights": trial, "rrf_k": rrf_k, "metrics": metrics}
                    if candidate_best is None or _better(result, candidate_best):
                        candidate_best = result
                if candidate_best and candidate_best["weights"][channel] != current[channel]:
                    current = candidate_best["weights"]
                    improved = True
        result = _evaluate(channel_rankings, questions, current, rrf_k, limit)
        if best is None or _better(result, best):
            best = result
    assert best is not None
    return best


def _evaluate(channel_rankings: dict[str, dict[str, list[str]]], questions: list[dict], weights: dict[str, float], rrf_k: int, limit: int) -> dict:
    predictions = {
        item["qid"]: rrf(
            {name: ranks[item["qid"]] for name, ranks in channel_rankings.items()}, weights, rrf_k, limit
        )
        for item in questions
    }
    return {"weights": dict(weights), "rrf_k": rrf_k, "metrics": score_predictions(predictions, questions)}


def _better(left: dict, right: dict) -> bool:
    return (left["metrics"]["recall"], left["metrics"]["precision"]) > (
        right["metrics"]["recall"],
        right["metrics"]["precision"],
    )
