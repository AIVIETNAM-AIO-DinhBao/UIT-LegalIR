from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path
from typing import Any

from .embeddings import audit_models, build_dense_index, build_question_embeddings
from .fusion import coordinate_search, rrf
from .lexical import LexicalIndex
from .prepare import build_corpus
from .rerank import final_predictions, rerank_candidates
from .retrieval import RetrievalPipeline
from .storage import read_json, read_jsonl, write_json
from .validation import grouped_folds, oracle_recall, score_candidates, score_predictions, validate_submission_shape


def prepare(config: dict[str, Any], resume: bool) -> dict[str, int]:
    return build_corpus(config, resume)


def index(config: dict[str, Any], resume: bool, model_name: str | None = None, lexical_only: bool = False) -> None:
    artifacts = Path(config["paths"]["artifacts_dir"])
    lexical_path = artifacts / "lexical_short.pkl"

    def build_lexical_index() -> None:
        chunks_path = artifacts / "chunks_short.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing {chunks_path}. Run the prepare command before indexing.")
        chunks = list(read_jsonl(chunks_path))
        if not chunks:
            raise ValueError(f"{chunks_path} contains no chunks. Re-run prepare with a non-empty contexts_dir.")
        LexicalIndex.build([chunk["text"] for chunk in chunks]).save(lexical_path)

    if (model_name is None or lexical_only) and not (resume and lexical_path.exists()):
        build_lexical_index()
    if lexical_only:
        return
    selected = {model_name} if model_name else None
    for candidate_name, spec in config["models"].items():
        if spec["role"] == "dense":
            if selected is not None and candidate_name not in selected:
                continue
            build_dense_index(config, candidate_name, resume)
            build_question_embeddings(config, candidate_name, resume)


def _load_questions(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    filename = "train_questions.jsonl" if split == "train" else "public_questions.jsonl"
    return list(read_jsonl(Path(config["paths"]["artifacts_dir"]) / filename))


def _train_allowed_questions(questions: list[dict[str, Any]], folds: int) -> dict[str, set[str]]:
    assignment = grouped_folds(questions, folds)
    all_qids = {item["qid"] for item in questions}
    return {qid: {other for other in all_qids if assignment[other] != assignment[qid]} for qid in all_qids}


def build_retrieval_cache(config: dict[str, Any], split: str, resume: bool) -> dict[str, dict[str, Any]]:
    artifacts = Path(config["paths"]["artifacts_dir"])
    cache = artifacts / f"retrieval_{split}.json"
    if resume and cache.exists():
        return read_json(cache)
    questions = _load_questions(config, split)
    allowed = _train_allowed_questions(questions, config["validation"]["folds"]) if split == "train" else None
    pipeline = RetrievalPipeline(config)
    result = pipeline.retrieve_many(questions, allowed)
    write_json(cache, result)
    return result


def tune_first_stage(config: dict[str, Any], resume: bool) -> dict[str, Any]:
    artifacts = Path(config["paths"]["artifacts_dir"])
    destination = artifacts / "first_stage_weights.json"
    if resume and destination.exists():
        return read_json(destination)
    questions = _load_questions(config, "train")
    retrievals = build_retrieval_cache(config, "train", resume)
    channel_rankings = {
        channel: {qid: retrievals[qid]["channels"][channel] for qid in retrievals}
        for channel in next(iter(retrievals.values()))["channels"]
    }
    options = config["retrieval"]
    tuning_limit = min(len(questions), options.get("first_stage_tuning_questions", len(questions)))
    tuning_questions = sorted(
        questions,
        key=lambda item: hashlib.sha256(item["qid"].encode("utf-8")).hexdigest(),
    )[:tuning_limit]
    best = coordinate_search(
        channel_rankings,
        tuning_questions,
        options["rrf_k_values"],
        options["first_stage_weight_values"],
        options["fused_top_k"],
        max_passes=options.get("coordinate_search_passes", 2),
    )
    # Exact normalized matches cannot appear in OOF training (duplicates share
    # a fold), but can legitimately occur in the public set.  Keep this
    # deterministic, high-confidence channel enabled instead of letting its
    # all-empty OOF ranking be tuned to zero.
    best["weights"]["query_exact"] = options["query_exact_weight"]
    best["tuning_questions"] = tuning_limit
    fused = _fuse_cached(config, retrievals, best["weights"], best["rrf_k"])
    candidates = {qid: value["candidates"] for qid, value in fused.items()}
    best["oracle_recall_at_fused_top_k"] = oracle_recall(candidates, questions)
    best["candidate_metrics"] = score_candidates(candidates, questions)
    best["channel_candidate_metrics"] = {
        channel: score_candidates(rankings, questions) for channel, rankings in channel_rankings.items()
    }
    write_json(destination, best)
    write_json(artifacts / "fused_train.json", fused)
    return best


def _fuse_cached(config: dict[str, Any], retrievals: dict[str, dict[str, Any]], weights: dict[str, float], rrf_k: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for qid, retrieval in retrievals.items():
        candidates = rrf(retrieval["channels"], weights, rrf_k, config["retrieval"]["fused_top_k"])
        evidence: dict[str, list[str]] = {document_id: [] for document_id in candidates}
        dense_channels = [name for name, spec in config["models"].items() if spec["role"] == "dense"]
        priority = [*dense_channels, "bm25", "accent_char"]
        priority.extend(name for name in retrieval["evidence"] if name not in priority)
        for channel in priority:
            per_document = retrieval["evidence"].get(channel, {})
            for document_id in candidates:
                evidence[document_id].extend(per_document.get(document_id, []))
        result[qid] = {
            "candidates": candidates,
            "evidence": {
                document_id: list(dict.fromkeys(chunk_ids))[: config["reranking"]["evidence_chunks_per_document"]]
                for document_id, chunk_ids in evidence.items()
            },
        }
    return result


def run_reranking(
    config: dict[str, Any], split: str, fold: int | None, resume: bool, engine: str | None = None
) -> dict[str, dict[str, list[str]]]:
    artifacts = Path(config["paths"]["artifacts_dir"])
    suffix = f"_{fold}" if fold is not None else ""
    destination = artifacts / f"rerank_{split}{suffix}.json"
    if resume and destination.exists():
        return read_json(destination)
    first_stage = read_json(artifacts / "first_stage_weights.json")
    retrievals = build_retrieval_cache(config, split, resume)
    fused = _fuse_cached(config, retrievals, first_stage["weights"], first_stage["rrf_k"])
    questions = _load_questions(config, split)
    if fold is not None:
        assignments = grouped_folds(_load_questions(config, "train"), config["validation"]["folds"])
        questions = [item for item in questions if assignments[item["qid"]] == fold]
        fused = {item["qid"]: fused[item["qid"]] for item in questions}
    engines = [engine] if engine else ["vietnamese_reranker", "jina"]
    if engine is None:
        existing = {
            name: read_json(artifacts / f"rerank_{split}{suffix}_{name}.json")[name]
            for name in ("vietnamese_reranker", "jina")
            if (artifacts / f"rerank_{split}{suffix}_{name}.json").exists()
        }
        if len(existing) == 2:
            write_json(destination, existing)
            return existing
    result = rerank_candidates(config, questions, fused, engines=engines)
    for name, rankings in result.items():
        write_json(artifacts / f"rerank_{split}{suffix}_{name}.json", {name: rankings})
    if engine is not None:
        return result
    write_json(destination, result)
    return result


def tune_final_stage(config: dict[str, Any], fold: int | None, resume: bool) -> dict[str, Any]:
    artifacts = Path(config["paths"]["artifacts_dir"])
    destination = artifacts / (f"final_weights_fold{fold}.json" if fold is not None else "final_weights.json")
    if resume and destination.exists():
        return read_json(destination)
    first_stage = read_json(artifacts / "first_stage_weights.json")
    retrievals = build_retrieval_cache(config, "train", resume)
    fused = _fuse_cached(config, retrievals, first_stage["weights"], first_stage["rrf_k"])
    all_questions = _load_questions(config, "train")
    assignments = grouped_folds(all_questions, config["validation"]["folds"])
    if fold is None:
        held_out_folds = config["validation"].get(
            "reranker_tuning_folds", list(range(config["validation"]["folds"]))
        )
        # Rankings remain OOF because query-memory labels from each selected
        # fold are withheld.  The runtime profile may intentionally select a
        # subset of folds for expensive model reranking.
        questions = [item for item in all_questions if assignments[item["qid"]] in held_out_folds]
        rerankings: dict[str, dict[str, list[str]]] = {"jina": {}, "vietnamese_reranker": {}}
        for held_out_fold in held_out_folds:
            per_fold = run_reranking(config, "train", held_out_fold, resume)
            for name in rerankings:
                rerankings[name].update(per_fold[name])
    else:
        held_out_folds = [fold]
        questions = [item for item in all_questions if assignments[item["qid"]] == fold]
        rerankings = run_reranking(config, "train", fold, resume)
    options = config["reranking"]
    best: dict[str, Any] | None = None
    for jina_weight, vi_weight, first_weight, rrf_k in product(
        options["final_reranker_weight_values"],
        options["final_reranker_weight_values"],
        options["final_first_stage_weight_values"],
        options["final_rrf_k_values"],
    ):
        weights = {"jina": jina_weight, "vietnamese_reranker": vi_weight, "first_stage": first_weight}
        predictions = final_predictions(
            {item["qid"]: fused[item["qid"]]["candidates"] for item in questions}, rerankings, weights, rrf_k
        )
        candidate = {"weights": weights, "rrf_k": rrf_k, "metrics": score_predictions(predictions, questions)}
        if best is None or candidate["metrics"]["recall"] > best["metrics"]["recall"]:
            best = candidate
    assert best is not None
    write_json(destination, best)
    if fold is None:
        per_fold_metrics: dict[str, dict[str, float]] = {}
        for held_out_fold in held_out_folds:
            fold_questions = [item for item in all_questions if assignments[item["qid"]] == held_out_fold]
            fold_predictions = final_predictions(
                {item["qid"]: fused[item["qid"]]["candidates"] for item in fold_questions},
                rerankings,
                best["weights"],
                best["rrf_k"],
            )
            per_fold_metrics[str(held_out_fold)] = score_predictions(fold_predictions, fold_questions)
        best["oof_fold_metrics"] = per_fold_metrics
        write_json(destination, best)
    return best


def predict(config: dict[str, Any], resume: bool, output: str | None = None) -> Path:
    artifacts = Path(config["paths"]["artifacts_dir"])
    first_stage = read_json(artifacts / "first_stage_weights.json")
    final_stage = read_json(artifacts / "final_weights.json")
    retrievals = build_retrieval_cache(config, "public", resume)
    fused = _fuse_cached(config, retrievals, first_stage["weights"], first_stage["rrf_k"])
    write_json(artifacts / "fused_public.json", fused)
    questions = _load_questions(config, "public")
    rerankings = run_reranking(config, "public", None, resume)
    predictions = final_predictions(
        {qid: value["candidates"] for qid, value in fused.items()}, rerankings, final_stage["weights"], final_stage["rrf_k"]
    )
    corpus_ids = {row["doc_id"] for row in read_jsonl(artifacts / "corpus.jsonl")}
    submission = {qid: {"answer": answer} for qid, answer in predictions.items()}
    validate_submission_shape(submission, questions, corpus_ids)
    destination = Path(output or config["paths"]["submission_file"])
    write_json(destination, submission)
    return destination


__all__ = ["audit_models", "build_retrieval_cache", "index", "predict", "prepare", "run_reranking", "tune_final_stage", "tune_first_stage"]
