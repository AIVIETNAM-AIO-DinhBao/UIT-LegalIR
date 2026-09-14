from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .fusion import rrf
from .storage import read_jsonl
from .text import tokenize_words


class EvidenceStore:
    def __init__(self, artifacts_dir: str | Path):
        self.chunks: dict[str, dict[str, Any]] = {}
        self.by_document: defaultdict[str, list[str]] = defaultdict(list)
        for path in (Path(artifacts_dir) / "chunks_short.jsonl", Path(artifacts_dir) / "chunks_long.jsonl"):
            for row in read_jsonl(path):
                self.chunks[row["chunk_id"]] = row
                self.by_document[row["doc_id"]].append(row["chunk_id"])

    def evidence(self, document_id: str, preferred: list[str], max_words: int) -> str:
        chunk_ids = list(dict.fromkeys(preferred + self.by_document.get(document_id, [])[:2]))
        text = "\n\n".join(self.chunks[chunk_id]["text"] for chunk_id in chunk_ids if chunk_id in self.chunks)
        return " ".join(tokenize_words(text)[:max_words])


def _device(runtime: dict[str, Any]) -> str:
    return runtime["device"] if torch.cuda.is_available() else "cpu"


class VietnamesePairwiseReranker:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        spec = config["models"]["vietnamese_reranker"]
        self.device = _device(config["runtime"])
        self.tokenizer = AutoTokenizer.from_pretrained(spec["id"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec["id"], torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
        ).to(self.device).eval()

    @torch.inference_mode()
    def rank(self, query: str, documents: list[str]) -> list[int]:
        batch_size = self.config["reranking"]["pairwise_batch_size"]
        maximum = self.config["reranking"]["pairwise_max_length"]
        scores: list[float] = []
        for start in range(0, len(documents), batch_size):
            pairs = [[query, document] for document in documents[start : start + batch_size]]
            inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=maximum, return_tensors="pt").to(self.device)
            scores.extend(self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist())
        return sorted(range(len(documents)), key=lambda index: (-scores[index], index))

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class JinaListwiseReranker:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        spec = config["models"]["jina"]
        self.device = _device(config["runtime"])
        self.model = AutoModel.from_pretrained(
            spec["id"],
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
        ).to(self.device).eval()

    @torch.inference_mode()
    def rank(self, query: str, documents: list[str]) -> list[int]:
        window = self.config["reranking"]["jina_window_size"]
        try:
            return self._rank_windows(query, documents, window)
        except torch.OutOfMemoryError:
            if not torch.cuda.is_available() or window <= 10:
                raise
            torch.cuda.empty_cache()
            return self._rank_windows(query, documents, 10)

    def _rank_windows(self, query: str, documents: list[str], window: int) -> list[int]:
        # Round-robin windows make every window contain candidates from different first-stage ranks.
        buckets = [[] for _ in range(max(1, (len(documents) + window - 1) // window))]
        for index, document in enumerate(documents):
            buckets[index % len(buckets)].append((index, document))
        scores: dict[int, float] = {}
        for bucket in buckets:
            results = self.model.rerank(query, [document for _, document in bucket])
            for result in results:
                scores[bucket[result["index"]][0]] = float(result["relevance_score"])
        semifinalists = [index for index, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: self.config["reranking"]["jina_final_top_k"]]]
        final_documents = [documents[index] for index in semifinalists]
        final = self.model.rerank(query, final_documents)
        final_indices = [semifinalists[result["index"]] for result in final]
        remaining = [index for index in range(len(documents)) if index not in set(final_indices)]
        return final_indices + sorted(remaining, key=lambda index: (-scores.get(index, float("-inf")), index))

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rerank_candidates(
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    fused: dict[str, dict[str, Any]],
    engines: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Run both approved rerankers locally and return cacheable document rankings."""
    store = EvidenceStore(config["paths"]["artifacts_dir"])
    engines = engines or ["vietnamese_reranker", "jina"]
    result: dict[str, dict[str, list[str]]] = {}
    if "vietnamese_reranker" in engines:
        pairwise = VietnamesePairwiseReranker(config)
        pairwise_rankings: dict[str, list[str]] = {}
        for item in questions:
            candidate = fused[item["qid"]]
            documents = [store.evidence(doc_id, candidate["evidence"].get(doc_id, []), 1400) for doc_id in candidate["candidates"]]
            order = pairwise.rank(item["question"], documents)
            pairwise_rankings[item["qid"]] = [candidate["candidates"][index] for index in order]
        pairwise.close()
        result["vietnamese_reranker"] = pairwise_rankings
    if "jina" in engines:
        jina = JinaListwiseReranker(config)
        jina_rankings: dict[str, list[str]] = {}
        for item in questions:
            candidate = fused[item["qid"]]
            documents = [
                store.evidence(doc_id, candidate["evidence"].get(doc_id, []), config["reranking"]["jina_evidence_tokens"])
                for doc_id in candidate["candidates"]
            ]
            order = jina.rank(item["question"], documents)
            jina_rankings[item["qid"]] = [candidate["candidates"][index] for index in order]
        jina.close()
        result["jina"] = jina_rankings
    return result


def final_predictions(
    first_stage: dict[str, list[str]],
    rerankings: dict[str, dict[str, list[str]]],
    weights: dict[str, float],
    rrf_k: int,
) -> dict[str, list[str]]:
    return {
        qid: rrf(
            {
                "first_stage": first_stage[qid],
                "jina": rerankings["jina"][qid],
                "vietnamese_reranker": rerankings["vietnamese_reranker"][qid],
            },
            weights,
            rrf_k,
            5,
        )
        for qid in first_stage
    }
