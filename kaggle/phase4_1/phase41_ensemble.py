"""One cache-only Phase 4.1 candidate; never runs retrieval or reranking."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

from legalir.storage import read_json, read_jsonl, write_json
from legalir.text import clean_text
from legalir.validation import grouped_folds, score_candidates, score_predictions, validate_submission_shape
from phase4_blend import (
    RERANKERS,
    baseline_predictions,
    engine_ranks,
    fuse_cached,
    inner_fold,
    make_rows,
    new_model,
    protected_predictions,
    standardize_per_query,
)


CS = (0.03, 0.1, 0.3)
REPORT_NAME = "phase41_report.json"
SUBMISSION_NAME = "submission_phase41_private_ensemble.json"


def required(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required cache/output missing or broken symlink: {path}")
    return path


def verify_source(report: dict, private_path: Path, phase4_dir: Path, phase2_dir: Path) -> None:
    content = required(private_path).read_bytes()
    if (report.get("experiment_id") != "phase4-phase2-legal-bge-protected-blender"
            or report.get("test_file") != "private-official.json"
            or report.get("test_sha256") != hashlib.sha256(content).hexdigest()
            or report.get("test_questions") != len(json.loads(content))):
        raise RuntimeError("Phase 4 report does not match the official private questions")
    p4 = read_json(required(phase4_dir / "artifacts_phase4" / "prepare_manifest.json"))
    p2 = read_json(required(phase2_dir / "prepare_manifest.json"))
    for key in ("schema_version", "chunking_fingerprint", "documents", "short_chunks", "long_chunks",
                "train_questions", "public_questions", "train_questions_fingerprint", "public_questions_fingerprint"):
        if p4.get(key) != p2.get(key):
            raise RuntimeError(f"Phase 2/4 prepared data differs: {key}")
    # Fail before training when a saved output omitted any expensive cache.
    for name in ("first_stage_weights.json", "final_weights.json", "retrieval_train.json", "retrieval_public.json"):
        required(phase2_dir / name)
    for split, suffix in (("train", "_0"), ("public", "")):
        for engine in ("jina", "vietnamese_reranker"):
            required(phase2_dir / f"rerank_{split}{suffix}_{engine}.json")
        required(phase4_dir / "artifacts_phase4" / f"rerank_{split}{suffix}_legal_reranker.json")
    for name in ("train_questions.jsonl", "public_questions.jsonl", "corpus.jsonl"):
        required(phase4_dir / "artifacts_phase4" / name)
    for name in ("kaggle_rtx_pro_6000_phase4.yaml", "submission_phase4_private_final.json"):
        required(phase4_dir / name)


def per_inner(predictions: dict[str, list[str]], questions: list[dict]) -> dict[str, dict[str, float]]:
    return {
        str(fold): score_predictions(
            predictions, [q for q in questions if inner_fold(q["question"]) == fold]
        ) for fold in range(5)
    }


def gate(ensemble: dict, phase4: dict, tolerance: float = 1e-12) -> tuple[bool, str]:
    if ensemble["metrics"]["recall"] <= phase4["metrics"]["recall"] + tolerance:
        return False, "Ensemble OOF Recall did not exceed Phase 4"
    for fold in range(5):
        key = str(fold)
        if ensemble["per_inner"][key]["recall"] < phase4["per_inner"][key]["recall"] - tolerance:
            return False, f"Ensemble OOF Recall declined in inner fold {fold}"
    return True, "OOF Recall improved with no declining inner fold"


def oof_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray, metadata: list[tuple[str, str]],
               c_values: tuple[float, ...] = CS) -> dict[float, np.ndarray]:
    scores = {}
    for c in c_values:
        raw = np.empty(len(y), dtype=np.float64)
        for fold in range(5):
            train, valid = groups != fold, groups == fold
            if not valid.any() or len(np.unique(y[train])) != 2:
                raise RuntimeError(f"Missing inner fold or class: {fold}")
            model = new_model(c)
            model.fit(x[train], y[train])
            raw[valid] = model.decision_function(x[valid])
        scores[c] = standardize_per_query(raw, metadata)
    return scores


def compare_phase4(predictions: dict[str, list[str]], saved_submission: dict) -> int:
    if set(predictions) != set(saved_submission):
        raise RuntimeError("Saved Phase 4 submission QIDs differ from reconstructed QIDs")
    changed = sum(set(documents) != set(saved_submission[qid]["answer"])
                  for qid, documents in predictions.items())
    return changed


def main(phase4_dir: Path, phase2_dir: Path, work: Path, private_path: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    # A rerun must never leave an old report or submission masquerading as a new result.
    for name in (REPORT_NAME, SUBMISSION_NAME, SUBMISSION_NAME.replace(".json", ".zip")):
        stale = work / name
        if stale.exists():
            stale.unlink()
    report = read_json(required(phase4_dir / "phase4_report.json"))
    verify_source(report, private_path, phase4_dir, phase2_dir)
    config = yaml.safe_load(required(phase4_dir / "kaggle_rtx_pro_6000_phase4.yaml").read_text(encoding="utf-8"))
    if config["validation"]["folds"] != 5 or config["retrieval"]["fused_top_k"] < 80:
        raise RuntimeError("Phase 4 fold count or candidate depth differs")
    p4_artifacts = phase4_dir / "artifacts_phase4"
    retrieval_train = read_json(phase2_dir / "retrieval_train.json")
    retrieval_private = read_json(phase2_dir / "retrieval_public.json")
    first_stage = read_json(phase2_dir / "first_stage_weights.json")
    weights = read_json(phase2_dir / "final_weights.json")
    fused_train = fuse_cached(retrieval_train, first_stage["weights"], first_stage["rrf_k"],
                              config["retrieval"]["fused_top_k"])
    fused_private = fuse_cached(retrieval_private, first_stage["weights"], first_stage["rrf_k"],
                                config["retrieval"]["fused_top_k"])
    train_questions = list(read_jsonl(p4_artifacts / "train_questions.jsonl"))
    private_questions = list(read_jsonl(p4_artifacts / "public_questions.jsonl"))
    official_private = json.loads(private_path.read_text(encoding="utf-8"))
    if len(private_questions) != report["test_questions"] or len(official_private) != len(private_questions):
        raise RuntimeError("Saved prepared private question count differs")
    # Prepared questions and raw official input must have the same IDs and text.
    if not isinstance(official_private, dict):
        raise RuntimeError("Official private file must be a QID-to-question mapping")
    if {q["qid"] for q in private_questions} != set(official_private) or any(
        clean_text(official_private[q["qid"]]["question"]) != q["question"] for q in private_questions
    ):
        raise RuntimeError("Saved prepared private questions do not match official input")
    fold_questions = [q for q in train_questions if grouped_folds([q], 5)[q["qid"]] == 0]
    train_ranks = {name: engine_ranks(phase2_dir if name != "legal_reranker" else p4_artifacts,
                                     "train", name, 0) for name in RERANKERS}
    private_ranks = {name: engine_ranks(phase2_dir if name != "legal_reranker" else p4_artifacts,
                                       "public", name) for name in RERANKERS}
    for questions, retrieval, fused, ranks in (
        (fold_questions, retrieval_train, fused_train, train_ranks),
        (private_questions, retrieval_private, fused_private, private_ranks),
    ):
        for q in questions:
            qid = q["qid"]
            candidates = fused[qid]["candidates"][:80]
            if len(candidates) != 80 or len(set(candidates)) != 80:
                raise RuntimeError(f"Invalid Phase 4 top 80: {qid}")
            for name in RERANKERS:
                ranking = ranks[name][qid]
                if len(ranking) != 80 or set(ranking) != set(candidates):
                    raise RuntimeError(f"Phase 4 {name} ranking/top-80 mismatch: {qid}")
            for name in ("bm25", "accent_char", "vietlegal_harrier", "vietnamese_embedding",
                         "nemotron", "query_memory", "query_exact"):
                if name not in retrieval[qid]["channels"]:
                    raise RuntimeError(f"Missing retrieval channel {name}: {qid}")

    baseline = baseline_predictions(fold_questions, fused_train, train_ranks, weights)
    if abs(score_predictions(baseline, fold_questions)["recall"] -
           report["phase2_baseline_fold0"]["recall"]) > 1e-10:
        raise RuntimeError("Phase 2 baseline cannot be reproduced from caches")
    x, y, metadata, groups = make_rows(fold_questions, fused_train, retrieval_train, train_ranks, True)
    selected = report["selected_cross_fitted"]
    if (selected["C"], selected["alpha"], selected["promotion_margin"]) != (0.1, 1.0, 0.0):
        raise RuntimeError("Saved Phase 4 selected a different configuration")
    if (len(fold_questions) != report["training_questions"] or len(y) != report["training_pairs"]
            or int(y.sum()) != report["positive_pairs"] or x.shape[1] != report["feature_count"]
            or score_candidates({q["qid"]: fused_train[q["qid"]]["candidates"][:80]
                                 for q in fold_questions}, fold_questions)["candidate_recall"]
            != report["candidate_recall_fold0"]):
        raise RuntimeError("Reconstructed Phase 4 training data differs from report")
    scores = oof_scores(x, y, groups, metadata)
    phase4_pred, _ = protected_predictions(scores[0.1], metadata, baseline, 0.0)
    ensemble_pred, promotions = protected_predictions(np.mean(list(scores.values()), axis=0), metadata, baseline, 0.0)
    phase4_oof = {"metrics": score_predictions(phase4_pred, fold_questions),
                  "per_inner": per_inner(phase4_pred, fold_questions)}
    ensemble_oof = {"metrics": score_predictions(ensemble_pred, fold_questions),
                    "per_inner": per_inner(ensemble_pred, fold_questions), **promotions}
    if abs(phase4_oof["metrics"]["recall"] - selected["metrics"]["recall"]) > 1e-10 or any(
        abs(phase4_oof["per_inner"][str(i)]["recall"] - selected["per_inner"][str(i)]["recall"]) > 1e-10
        for i in range(5)
    ):
        raise RuntimeError("Phase 4 OOF cannot be reproduced; refusing to evaluate Phase 4.1")

    approved, reason = gate(ensemble_oof, phase4_oof)
    output = {"experiment_id": "phase41-cache-only-three-logistic-ensemble",
              "phase4_report_sha256": hashlib.sha256((phase4_dir / "phase4_report.json").read_bytes()).hexdigest(),
              "test_sha256": report["test_sha256"], "C_values": list(CS), "alpha": 1.0,
              "promotion_margin": 0.0, "phase4_oof": phase4_oof, "ensemble_oof": ensemble_oof,
              "oof_delta": ensemble_oof["metrics"]["recall"] - phase4_oof["metrics"]["recall"],
              "approved": approved, "reason": reason, "submission": None,
              "warning": "OOF selection is not an unbiased estimate of private Recall."}
    if not approved:
        write_json(work / REPORT_NAME, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    x_private, _, private_meta, _ = make_rows(private_questions, fused_private,
                                               retrieval_private, private_ranks, False)
    if x_private.shape[1] != x.shape[1]:
        raise RuntimeError("Private feature dimension differs from training")
    full_scores = []
    for c in CS:
        model = new_model(c)
        model.fit(x, y)
        full_scores.append(standardize_per_query(model.decision_function(x_private), private_meta))
    private_baseline = baseline_predictions(private_questions, fused_private, private_ranks, weights)
    phase4_full = new_model(0.1)
    phase4_full.fit(x, y)
    reconstructed, _ = protected_predictions(
        standardize_per_query(phase4_full.decision_function(x_private), private_meta),
        private_meta, private_baseline, 0.0,
    )
    saved = read_json(phase4_dir / "submission_phase4_private_final.json")
    if compare_phase4(reconstructed, saved):
        raise RuntimeError("Reconstructed Phase 4 private answer sets differ from saved submission")
    predictions, private_promotions = protected_predictions(
        np.mean(full_scores, axis=0), private_meta, private_baseline, 0.0
    )
    changed = compare_phase4(predictions, saved)
    if changed == 0:
        output["approved"] = False
        output["reason"] = "Ensemble is identical to Phase 4 on private inputs; do not spend last submission"
        output["changed_answer_sets_vs_phase4"] = 0
        write_json(work / REPORT_NAME, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    corpus_ids = {row["doc_id"] for row in read_jsonl(p4_artifacts / "corpus.jsonl")}
    submission = {qid: {"answer": docs} for qid, docs in predictions.items()}
    validate_submission_shape(submission, private_questions, corpus_ids)
    output["private_promotions"] = private_promotions
    output["changed_answer_sets_vs_phase4"] = changed
    output["submission"] = str(work / SUBMISSION_NAME)
    output["zip"] = str(work / SUBMISSION_NAME.replace(".json", ".zip"))
    write_json(work / SUBMISSION_NAME, submission)
    with zipfile.ZipFile(output["zip"], "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(work / SUBMISSION_NAME, arcname=SUBMISSION_NAME)
    write_json(work / REPORT_NAME, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("Usage: phase41_ensemble.py PHASE4_OUTPUT PHASE2_ARTIFACTS WORK_DIR PRIVATE_JSON")
    main(*(Path(value) for value in sys.argv[1:]))