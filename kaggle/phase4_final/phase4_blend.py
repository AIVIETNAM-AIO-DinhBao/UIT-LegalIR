from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from legalir.fusion import rrf
from legalir.storage import read_json, read_jsonl, write_json
from legalir.text import normalize_question
from legalir.validation import grouped_folds, score_candidates, score_predictions, validate_submission_shape


RERANKERS = ("jina", "vietnamese_reranker", "legal_reranker")
RETRIEVAL_CHANNELS = (
    "bm25",
    "accent_char",
    "vietlegal_harrier",
    "vietnamese_embedding",
    "nemotron",
    "query_memory",
    "query_exact",
)


def fuse_cached(
    retrievals: dict[str, dict[str, Any]],
    weights: dict[str, float],
    rrf_k: int,
    limit: int,
) -> dict[str, dict[str, list[str]]]:
    return {
        qid: {"candidates": rrf(retrieval["channels"], weights, rrf_k, limit)}
        for qid, retrieval in retrievals.items()
    }


def engine_ranks(artifacts: Path, split: str, engine: str, fold: int | None = None) -> dict[str, list[str]]:
    suffix = f"_{fold}" if fold is not None else ""
    separate = artifacts / f"rerank_{split}{suffix}_{engine}.json"
    if separate.is_file():
        return read_json(separate)[engine]
    combined = artifacts / f"rerank_{split}{suffix}.json"
    payload = read_json(combined)
    if engine not in payload:
        raise RuntimeError(f"{engine} is missing from {combined}")
    return payload[engine]


def inner_fold(question: str) -> int:
    # A salt different from grouped_folds is essential: every selected question
    # is already in outer fold zero under the main fold hash.
    key = "phase4-inner-v1:" + normalize_question(question)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) % 5


def reciprocal_rank(rank: int, k: int = 20) -> float:
    return 1.0 / (k + rank)


def rank_features(rank: int, missing_rank: int) -> list[float]:
    clipped = min(rank, missing_rank)
    denominator = max(1, missing_rank - 1)
    return [
        reciprocal_rank(clipped),
        1.0 / clipped,
        1.0 - min(clipped - 1, denominator) / denominator,
        float(clipped <= 5),
        float(clipped <= 10),
        float(clipped <= 20),
        float(clipped <= 50),
        float(clipped < missing_rank),
    ]


def feature_vector(document: str, orders: dict[str, dict[str, int]]) -> list[float]:
    features: list[float] = []
    core_ranks: list[int] = []
    for name in ("first_stage", *RERANKERS):
        rank = orders[name].get(document, 81)
        core_ranks.append(rank)
        features.extend(rank_features(rank, 81))
    retrieval_ranks: list[int] = []
    for name in RETRIEVAL_CHANNELS:
        rank = orders[name].get(document, 151)
        retrieval_ranks.append(rank)
        features.extend(rank_features(rank, 151))

    all_ranks = core_ranks + retrieval_ranks
    features.extend(
        [
            float(sum(rank <= 5 for rank in core_ranks)),
            float(sum(rank <= 10 for rank in core_ranks)),
            float(sum(rank <= 20 for rank in core_ranks)),
            float(sum(rank <= 20 for rank in retrieval_ranks)),
            float(sum(rank < 151 for rank in retrieval_ranks)),
            float(min(all_ranks)),
            float(max(core_ranks)),
            float(np.mean(core_ranks)),
            float(np.std(core_ranks)),
        ]
    )
    phase2_score = (
        0.3 * reciprocal_rank(core_ranks[0])
        + 0.5 * reciprocal_rank(core_ranks[1])
        + 0.5 * reciprocal_rank(core_ranks[2])
    )
    legal_residual = reciprocal_rank(core_ranks[3]) - reciprocal_rank(core_ranks[0])
    features.extend([phase2_score, legal_residual])
    return features


def make_rows(
    questions: list[dict[str, Any]],
    fused: dict[str, dict[str, Any]],
    retrievals: dict[str, dict[str, Any]],
    rerankings: dict[str, dict[str, list[str]]],
    labelled: bool,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    metadata: list[tuple[str, str]] = []
    groups: list[int] = []
    for question in questions:
        qid = question["qid"]
        candidates = fused[qid]["candidates"][:80]
        orders = {"first_stage": {doc: rank for rank, doc in enumerate(candidates, 1)}}
        for name, values in rerankings.items():
            orders[name] = {doc: rank for rank, doc in enumerate(values[qid], 1)}
        for name in RETRIEVAL_CHANNELS:
            orders[name] = {doc: rank for rank, doc in enumerate(retrievals[qid]["channels"][name], 1)}
        truth = set(question.get("answers", []))
        group = inner_fold(question["question"])
        for document in candidates:
            rows.append(feature_vector(document, orders))
            labels.append(int(document in truth) if labelled else 0)
            metadata.append((qid, document))
            groups.append(group)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        metadata,
        np.asarray(groups, dtype=np.int8),
    )


def standardize_per_query(values: np.ndarray, metadata: list[tuple[str, str]]) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float64)
    by_qid: defaultdict[str, list[int]] = defaultdict(list)
    for index, (qid, _) in enumerate(metadata):
        by_qid[qid].append(index)
    for indices in by_qid.values():
        scores = values[indices]
        scale = scores.std()
        result[indices] = (scores - scores.mean()) / (scale if scale > 1e-9 else 1.0)
    return result


def scores_by_question(values: np.ndarray, metadata: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    output: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for score, (qid, document) in zip(values, metadata, strict=True):
        output[qid][document] = float(score)
    return dict(output)


def protected_predictions(
    blended_scores: np.ndarray,
    metadata: list[tuple[str, str]],
    baseline: dict[str, list[str]],
    margin: float,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    per_query = scores_by_question(blended_scores, metadata)
    output: dict[str, list[str]] = {}
    promoted_questions = 0
    promotions = 0
    for qid, scores in per_query.items():
        selected = list(baseline[qid])
        outsiders = [doc for doc, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if doc not in selected]
        changed = False
        for outsider in outsiders:
            weakest = min(selected, key=lambda doc: (scores[doc], doc))
            if scores[outsider] < scores[weakest] + margin:
                break
            selected[selected.index(weakest)] = outsider
            promotions += 1
            changed = True
        if changed:
            promoted_questions += 1
        # Ordering is irrelevant to Recall, but sorting makes the output deterministic.
        output[qid] = sorted(selected, key=lambda doc: (-scores[doc], doc))
    return output, {"promoted_questions": promoted_questions, "promotions": promotions}


def new_model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=600,
            solver="liblinear",
            random_state=2026,
        ),
    )


def baseline_predictions(
    questions: list[dict[str, Any]],
    fused: dict[str, dict[str, Any]],
    rerankings: dict[str, dict[str, list[str]]],
    final_weights: dict[str, Any],
) -> dict[str, list[str]]:
    return {
        question["qid"]: rrf(
            {
                "first_stage": fused[question["qid"]]["candidates"],
                "jina": rerankings["jina"][question["qid"]],
                "vietnamese_reranker": rerankings["vietnamese_reranker"][question["qid"]],
            },
            final_weights["weights"],
            final_weights["rrf_k"],
            5,
        )
        for question in questions
    }


def main() -> None:
    work = Path(sys.argv[1])
    config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
    artifacts = Path(config["paths"]["artifacts_dir"])
    if not artifacts.is_absolute():
        artifacts = work / artifacts

    first_stage = read_json(artifacts / "first_stage_weights.json")
    phase2_final = read_json(artifacts / "phase2_final_weights.json")
    retrieval_train = read_json(artifacts / "retrieval_train.json")
    retrieval_public = read_json(artifacts / "retrieval_public.json")
    fused_limit = int(config["retrieval"]["fused_top_k"])
    fused_train = fuse_cached(retrieval_train, first_stage["weights"], first_stage["rrf_k"], fused_limit)
    fused_public = fuse_cached(retrieval_public, first_stage["weights"], first_stage["rrf_k"], fused_limit)
    train_questions = list(read_jsonl(artifacts / "train_questions.jsonl"))
    public_questions = list(read_jsonl(artifacts / "public_questions.jsonl"))
    outer_folds = grouped_folds(train_questions, config["validation"]["folds"])
    fold_questions = [question for question in train_questions if outer_folds[question["qid"]] == 0]

    train_rerankings = {name: engine_ranks(artifacts, "train", name, 0) for name in RERANKERS}
    public_rerankings = {name: engine_ranks(artifacts, "public", name) for name in RERANKERS}
    for questions, rerankings, split in (
        (fold_questions, train_rerankings, "train"),
        (public_questions, public_rerankings, "public"),
    ):
        expected = {question["qid"] for question in questions}
        for name, values in rerankings.items():
            missing = expected.difference(values)
            wrong_depth = [qid for qid in expected.intersection(values) if len(values[qid]) != 80]
            if missing or wrong_depth:
                raise RuntimeError(f"{split}/{name}: missing={len(missing)}, non_top80={len(wrong_depth)}")

    baseline = baseline_predictions(fold_questions, fused_train, train_rerankings, phase2_final)
    baseline_metrics = score_predictions(baseline, fold_questions)
    if any(not set(baseline[q["qid"]]).issubset(fused_train[q["qid"]]["candidates"][:80]) for q in fold_questions):
        raise RuntimeError("Phase 2 baseline selected a document outside top 80")

    x_train, y_train, train_metadata, inner_groups = make_rows(
        fold_questions, fused_train, retrieval_train, train_rerankings, True
    )
    if y_train.sum() == 0 or set(inner_groups) != set(range(5)):
        raise RuntimeError(
            f"Invalid training sample: positives={int(y_train.sum())}, inner_folds={sorted(set(inner_groups))}"
        )
    anchor_train = standardize_per_query(x_train[:, -2].astype(np.float64), train_metadata)
    baseline_per_inner = {
        str(fold): score_predictions(
            baseline,
            [question for question in fold_questions if inner_fold(question["question"]) == fold],
        )
        for fold in range(5)
    }

    trials: list[dict[str, Any]] = []
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros(len(y_train), dtype=np.float64)
        for heldout in range(5):
            train_mask = inner_groups != heldout
            valid_mask = inner_groups == heldout
            model = new_model(c_value)
            model.fit(x_train[train_mask], y_train[train_mask])
            oof[valid_mask] = model.decision_function(x_train[valid_mask])
        learned = standardize_per_query(oof, train_metadata)
        for alpha in (0.35, 0.5, 0.65, 0.8, 1.0):
            blended = alpha * learned + (1.0 - alpha) * anchor_train
            for margin in (0.0, 0.25, 0.5, 0.75, 1.0):
                prediction, promotion = protected_predictions(blended, train_metadata, baseline, margin)
                per_inner = {
                    str(fold): score_predictions(
                        prediction,
                        [question for question in fold_questions if inner_fold(question["question"]) == fold],
                    )
                    for fold in range(5)
                }
                deltas = [per_inner[str(fold)]["recall"] - baseline_per_inner[str(fold)]["recall"] for fold in range(5)]
                trials.append(
                    {
                        "C": c_value,
                        "alpha": alpha,
                        "promotion_margin": margin,
                        "metrics": score_predictions(prediction, fold_questions),
                        "per_inner": per_inner,
                        "nonnegative_inner_folds": sum(delta >= -1e-12 for delta in deltas),
                        "worst_inner_recall_delta": min(deltas),
                        **promotion,
                    }
                )

    robust = [
        row
        for row in trials
        if row["nonnegative_inner_folds"] >= 4 and row["worst_inner_recall_delta"] >= -0.005
    ]
    pool = robust or trials
    best = max(
        pool,
        key=lambda row: (
            row["metrics"]["recall"],
            row["nonnegative_inner_folds"],
            row["worst_inner_recall_delta"],
            row["metrics"]["precision"],
            -row["promotions"],
            -row["C"],
        ),
    )

    final_model = new_model(best["C"])
    final_model.fit(x_train, y_train)
    x_public, _, public_metadata, _ = make_rows(
        public_questions, fused_public, retrieval_public, public_rerankings, False
    )
    learned_public = standardize_per_query(final_model.decision_function(x_public), public_metadata)
    anchor_public = standardize_per_query(x_public[:, -2].astype(np.float64), public_metadata)
    baseline_public = baseline_predictions(public_questions, fused_public, public_rerankings, phase2_final)
    blended_public = best["alpha"] * learned_public + (1.0 - best["alpha"]) * anchor_public
    final_prediction, private_promotion = protected_predictions(
        blended_public, public_metadata, baseline_public, best["promotion_margin"]
    )

    corpus_ids = {row["doc_id"] for row in read_jsonl(artifacts / "corpus.jsonl")}
    submission = {qid: {"answer": documents} for qid, documents in final_prediction.items()}
    validate_submission_shape(submission, public_questions, corpus_ids)
    submission_path = work / "submission_phase4_private_final.json"
    write_json(submission_path, submission)
    with (work / "phase4_blender.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)

    report = {
        "experiment_id": "phase4-phase2-legal-bge-protected-blender",
        "training_questions": len(fold_questions),
        "training_pairs": len(y_train),
        "positive_pairs": int(y_train.sum()),
        "feature_count": int(x_train.shape[1]),
        "inner_fold_counts": {
            str(fold): int(sum(inner_fold(question["question"]) == fold for question in fold_questions))
            for fold in range(5)
        },
        "candidate_top_k": 80,
        "candidate_recall_fold0": score_candidates(
            {question["qid"]: fused_train[question["qid"]]["candidates"][:80] for question in fold_questions},
            fold_questions,
        )["candidate_recall"],
        "phase2_baseline_fold0": baseline_metrics,
        "phase2_baseline_per_inner": baseline_per_inner,
        "selected_cross_fitted": best,
        "recall_delta": best["metrics"]["recall"] - baseline_metrics["recall"],
        "robust_trials": len(robust),
        "tested_configurations": len(trials),
        "private_promotions": private_promotion,
        "submission": str(submission_path),
    }
    write_json(work / "phase4_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()