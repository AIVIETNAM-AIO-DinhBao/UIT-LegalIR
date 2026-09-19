from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, resolve_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline UIT DSC 2026 LegalIR pipeline")
    parser.add_argument("command", choices=["prepare", "audit", "index", "retrieve", "tune", "rerank", "predict"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--split", choices=["train", "public"], default="train")
    parser.add_argument("--final", action="store_true", help="Tune final reranker fusion after reranking")
    parser.add_argument("--output")
    parser.add_argument("--model", help="Dense model key from the selected configuration")
    parser.add_argument("--lexical-only", action="store_true")
    # Reranker names are configuration keys.  Keeping a hard-coded list here
    # made the Phase 3 engines impossible to invoke.  run_reranking validates
    # that the requested key exists and has a reranker role.
    parser.add_argument("--engine")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # Keep `python -m legalir --help` and lightweight source checks usable on a
    # machine that has not installed the Kaggle/GPU dependency set yet.
    from .pipeline import audit_models, index, predict, prepare, run_reranking, tune_final_stage, tune_first_stage

    config = resolve_paths(load_config(args.config), Path.cwd())

    if args.command == "prepare":
        result = prepare(config, args.resume)
    elif args.command == "audit":
        result = audit_models(config)
    elif args.command == "index":
        index(config, args.resume, args.model, args.lexical_only)
        result = {"status": "indexed"}
    elif args.command == "retrieve":
        from .pipeline import build_retrieval_cache

        result = {"questions": len(build_retrieval_cache(config, args.split, args.resume))}
    elif args.command == "tune":
        result = tune_final_stage(config, args.fold, args.resume) if args.final else tune_first_stage(config, args.resume)
    elif args.command == "rerank":
        result = run_reranking(config, args.split, args.fold if args.split == "train" else None, args.resume, args.engine)
    else:
        result = {"submission": str(predict(config, args.resume, args.output))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
