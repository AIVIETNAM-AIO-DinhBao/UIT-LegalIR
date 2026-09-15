from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .models import model_load_kwargs, model_source
from .storage import ensure_dir, read_jsonl, write_json


def _torch_dtype(name: str):
    import torch

    return torch.float16 if name == "float16" else torch.bfloat16


def load_encoder(model_spec: dict[str, Any], runtime: dict[str, Any]) -> SentenceTransformer:
    """Load a local checkpoint for direct inference; never route through a provider."""
    import torch

    device = runtime["device"] if torch.cuda.is_available() else "cpu"
    kwargs: dict[str, Any] = {"device": device}
    if device.startswith("cuda"):
        kwargs["model_kwargs"] = {
            "dtype": _torch_dtype(runtime["dtype"]),
            "attn_implementation": "sdpa",
        }
    kwargs.update(model_load_kwargs(model_spec))
    return SentenceTransformer(model_source(model_spec), **kwargs)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    description: str,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=description):
        encoded = model.encode(
            texts[start : start + batch_size],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors.append(np.asarray(encoded, dtype=np.float32))
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)


def build_dense_index(config: dict[str, Any], model_name: str, resume: bool = False) -> None:
    spec = config["models"][model_name]
    if spec["role"] != "dense":
        return
    artifacts = Path(config["paths"]["artifacts_dir"])
    destination = ensure_dir(artifacts / "dense" / model_name)
    vectors_path = destination / "vectors.npy"
    index_path = destination / "index.faiss"
    chunks_path = artifacts / ("chunks_short.jsonl" if spec["chunk_view"] == "short" else "chunks_long.jsonl")
    if resume and vectors_path.exists() and index_path.exists():
        return
    chunks = list(read_jsonl(chunks_path))
    texts = [spec["prompt_document"] + chunk["text"] for chunk in chunks]
    model = load_encoder(spec, config["runtime"])
    vectors = encode_texts(model, texts, config["runtime"]["batch_size"], f"Encoding {model_name}")
    np.save(vectors_path, vectors.astype(np.float16))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    write_json(destination / "chunks.json", chunks)
    del model, vectors, index
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def build_question_embeddings(config: dict[str, Any], model_name: str, resume: bool = False) -> None:
    spec = config["models"][model_name]
    if spec["role"] != "dense":
        return
    artifacts = Path(config["paths"]["artifacts_dir"])
    destination = ensure_dir(artifacts / "question_memory" / model_name)
    vector_path = destination / "vectors.npy"
    if resume and vector_path.exists():
        return
    questions = list(read_jsonl(artifacts / "train_questions.jsonl"))
    model = load_encoder(spec, config["runtime"])
    vectors = encode_texts(
        model,
        [spec["prompt_query"] + item["question"] for item in questions],
        config["runtime"]["batch_size"],
        f"Encoding train questions ({model_name})",
    )
    np.save(vector_path, vectors.astype(np.float16))
    write_json(destination / "questions.json", questions)
    del model, vectors
    gc.collect()


def audit_models(config: dict[str, Any]) -> dict[str, Any]:
    """Count checkpoint parameters locally, including every deployed component."""
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification

    rows: list[dict[str, Any]] = []
    total = 0
    for name, spec in config["models"].items():
        loader = AutoModelForSequenceClassification if spec["role"] == "pairwise_reranker" else AutoModel
        model = loader.from_pretrained(
            model_source(spec),
            trust_remote_code=name == "jina",
            dtype=torch.float16,
            **model_load_kwargs(spec),
        )
        count = sum(parameter.numel() for parameter in model.parameters())
        rows.append(
            {
                "name": name,
                "id": spec["id"],
                "parameters": count,
                "revision": spec.get("revision") or getattr(model.config, "_commit_hash", None),
                "license": spec.get("license", "not-recorded"),
            }
        )
        total += count
        del model
        gc.collect()
    manifest = {"models": rows, "total_parameters": total, "limit_parameters": 4_000_000_000, "compliant": total < 4_000_000_000}
    if not manifest["compliant"]:
        raise RuntimeError(f"Model system has {total:,} parameters, violating the 4B limit")
    write_json(Path(config["paths"]["artifacts_dir"]) / "model_manifest.json", manifest)
    return manifest
