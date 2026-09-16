from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # Keep corpus preparation usable in minimal test tooling.
    def tqdm(iterable, **_: Any):  # type: ignore[misc]
        return iterable

from .storage import ensure_dir, read_json, write_json, write_jsonl
from .text import clean_text, legal_chunks


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    raw = read_json(path)
    return [
        {
            "qid": str(qid),
            "question": clean_text(item["question"]),
            "answers": [str(answer) for answer in item.get("answer") or []],
        }
        for qid, item in raw.items()
    ]


def build_corpus(config: dict[str, Any], resume: bool = False) -> dict[str, int]:
    paths = config["paths"]
    artifacts = ensure_dir(paths["artifacts_dir"])
    corpus_path = artifacts / "corpus.jsonl"
    short_path = artifacts / "chunks_short.jsonl"
    long_path = artifacts / "chunks_long.jsonl"
    train_path = artifacts / "train_questions.jsonl"
    public_path = artifacts / "public_questions.jsonl"
    manifest_path = artifacts / "prepare_manifest.json"
    chunking_fingerprint = hashlib.sha256(
        json.dumps(config["chunking"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if resume and all(path.exists() for path in (corpus_path, short_path, long_path, train_path, public_path, manifest_path)):
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema_version") == 2
            and manifest.get("chunking_fingerprint") == chunking_fingerprint
            and manifest.get("documents", 0) > 0
            and manifest.get("short_chunks", 0) > 0
            and manifest.get("long_chunks", 0) > 0
        ):
            return manifest

    contexts_dir = Path(paths["contexts_dir"])
    context_files = sorted(
        path
        for path in contexts_dir.glob("context_*.json")
        if not path.name.endswith(".Zone.Identifier")
    )
    if not context_files:
        raise FileNotFoundError(
            f"No legal context files matching 'context_*.json' were found in {contexts_dir.resolve()}. "
            "Set paths.contexts_dir to the directory that directly contains the context files."
        )
    short_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    chunking = config["chunking"]
    for source in tqdm(context_files, desc="Preparing legal corpus"):
        with source.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        doc_id = str(raw["id"])
        name = clean_text(raw.get("name"))
        passage = clean_text(raw.get("passage"))
        corpus_rows.append({"doc_id": doc_id, "name": name, "link": raw.get("link", ""), "passage": passage})
        short, long = legal_chunks(
            name,
            passage,
            chunking["short_words"],
            chunking["short_overlap_words"],
            chunking["long_words"],
            chunking["long_overlap_words"],
        )
        for position, chunk in enumerate(short):
            short_rows.append({"chunk_id": f"{doc_id}:s:{position}", "doc_id": doc_id, **chunk})
        for position, chunk in enumerate(long):
            long_rows.append({"chunk_id": f"{doc_id}:l:{position}", "doc_id": doc_id, **chunk})

    write_jsonl(corpus_path, corpus_rows)
    write_jsonl(short_path, short_rows)
    write_jsonl(long_path, long_rows)
    write_jsonl(train_path, load_questions(paths["train_file"]))
    write_jsonl(public_path, load_questions(paths["public_file"]))
    manifest = {
        "schema_version": 2,
        "chunking_fingerprint": chunking_fingerprint,
        "documents": len(corpus_rows),
        "short_chunks": len(short_rows),
        "long_chunks": len(long_rows),
        "train_questions": len(load_questions(paths["train_file"])),
        "public_questions": len(load_questions(paths["public_file"])),
    }
    write_json(manifest_path, manifest)
    return manifest
