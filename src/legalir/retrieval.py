from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .embeddings import encode_texts, load_encoder
from .fusion import rrf
from .lexical import LexicalIndex
from .storage import read_json, read_jsonl
from .text import normalize_question


class RetrievalPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.artifacts = Path(config["paths"]["artifacts_dir"])
        self.short_chunks = list(read_jsonl(self.artifacts / "chunks_short.jsonl"))
        self.lexical = LexicalIndex.load(self.artifacts / "lexical_short.pkl")
        self.dense: dict[str, tuple[Any, list[dict[str, Any]]]] = {}
        self.question_vectors: dict[str, np.ndarray] = {}
        self.train_questions = list(read_jsonl(self.artifacts / "train_questions.jsonl"))
        self.train_by_normalized_question: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.train_questions:
            self.train_by_normalized_question[normalize_question(item["question"])].append(item)

    def _dense_index(self, model_name: str) -> tuple[Any, list[dict[str, Any]]]:
        if model_name not in self.dense:
            base = self.artifacts / "dense" / model_name
            self.dense[model_name] = (faiss.read_index(str(base / "index.faiss")), read_json(base / "chunks.json"))
        return self.dense[model_name]

    def _question_vectors(self, model_name: str) -> np.ndarray:
        if model_name not in self.question_vectors:
            self.question_vectors[model_name] = np.load(self.artifacts / "question_memory" / model_name / "vectors.npy").astype(np.float32)
        return self.question_vectors[model_name]

    def _search_vectors(
        self,
        model_name: str,
        index: Any,
        query_vectors: np.ndarray,
        k: int,
        memory: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact IP search on the active GPU, with FAISS CPU as a portable fallback."""
        try:
            import torch

            if torch.cuda.is_available() and self.config["runtime"]["device"].startswith("cuda"):
                if memory:
                    corpus = self._question_vectors(model_name)
                else:
                    corpus = np.load(self.artifacts / "dense" / model_name / "vectors.npy", mmap_mode="r")
                corpus_tensor = torch.from_numpy(np.asarray(corpus)).to("cuda", dtype=torch.float16)
                scores: list[np.ndarray] = []
                indices: list[np.ndarray] = []
                batch_size = self.config["runtime"]["batch_size"]
                for start in range(0, len(query_vectors), batch_size):
                    batch = torch.from_numpy(query_vectors[start : start + batch_size]).to("cuda", dtype=torch.float16)
                    values, positions = torch.topk(batch @ corpus_tensor.T, k=min(k, len(corpus_tensor)), dim=1)
                    scores.append(values.float().cpu().numpy())
                    indices.append(positions.cpu().numpy())
                del corpus_tensor
                torch.cuda.empty_cache()
                return np.concatenate(scores), np.concatenate(indices)
        except Exception as error:
            # CPU FAISS remains valid for local functional testing. The message
            # makes unexpected GPU failures visible without discarding a run.
            print(f"GPU exact search unavailable for {model_name}: {error}; falling back to FAISS CPU")
        if memory:
            memory_index = faiss.IndexFlatIP(self._question_vectors(model_name).shape[1])
            memory_index.add(self._question_vectors(model_name))
            scores, indices = memory_index.search(query_vectors.astype(np.float32), k)
            return scores, indices
        return index.search(query_vectors.astype(np.float32), k)

    @staticmethod
    def _aggregate_chunks(indices: np.ndarray, scores: np.ndarray, chunks: list[dict[str, Any]], top_k: int, second_weight: float) -> tuple[list[str], dict[str, list[str]]]:
        by_document: defaultdict[str, list[tuple[float, str]]] = defaultdict(list)
        for index, score in zip(indices.tolist(), scores.tolist(), strict=True):
            chunk = chunks[index]
            by_document[chunk["doc_id"]].append((float(score), chunk["chunk_id"]))
        aggregates = []
        evidence: dict[str, list[str]] = {}
        for doc_id, values in by_document.items():
            values.sort(reverse=True)
            aggregate = values[0][0] + (second_weight * values[1][0] if len(values) > 1 else 0.0)
            aggregates.append((doc_id, aggregate))
            evidence[doc_id] = [chunk_id for _, chunk_id in values[:2]]
        aggregates.sort(key=lambda item: (-item[1], item[0]))
        return [doc_id for doc_id, _ in aggregates[:top_k]], evidence

    def lexical_rankings(self, question: str) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
        output: dict[str, list[str]] = {}
        evidence: dict[str, dict[str, list[str]]] = {}
        options = self.config["retrieval"]
        for channel in ("bm25", "accent_char"):
            indices, scores = self.lexical.search(question, channel, options["chunk_search_k"])
            ranking, channel_evidence = self._aggregate_chunks(indices, scores, self.short_chunks, options["per_channel_top_k"], options["doc_second_chunk_weight"])
            output[channel] = ranking
            evidence[channel] = channel_evidence
        return output, evidence

    def retrieve_many(
        self,
        questions: list[dict[str, Any]],
        allowed_train_qids: dict[str, set[str]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Retrieve a batch while each dense checkpoint is resident only once."""
        options = self.config["retrieval"]
        output: dict[str, dict[str, Any]] = {
            item["qid"]: {"channels": {}, "evidence": {}} for item in questions
        }
        for item in questions:
            lexical, lexical_evidence = self.lexical_rankings(item["question"])
            output[item["qid"]]["channels"].update(lexical)
            output[item["qid"]]["evidence"].update(lexical_evidence)

        memory_scores: dict[str, defaultdict[str, float]] = {
            item["qid"]: defaultdict(float) for item in questions
        }
        memory_models: dict[str, defaultdict[str, set[str]]] = {
            item["qid"]: defaultdict(set) for item in questions
        }
        memory_first_rank: dict[str, dict[str, int]] = {item["qid"]: {} for item in questions}
        for model_name, spec in self.config["models"].items():
            if spec["role"] != "dense":
                continue
            index, chunks = self._dense_index(model_name)
            model = load_encoder(spec, self.config["runtime"])
            vectors = encode_texts(
                model,
                [spec["prompt_query"] + item["question"] for item in questions],
                self.config["runtime"]["batch_size"],
                f"Retrieving {model_name}",
            )
            scores, indices = self._search_vectors(model_name, index, vectors, options["chunk_search_k"])
            neighbour_k = min(len(self.train_questions), max(256, options["query_memory_neighbors"] * 8))
            memory_values, memory_indices = self._search_vectors(model_name, index, vectors, neighbour_k, memory=True)
            for row, item in enumerate(questions):
                ranking, channel_evidence = self._aggregate_chunks(
                    indices[row], scores[row], chunks, options["per_channel_top_k"], options["doc_second_chunk_weight"]
                )
                output[item["qid"]]["channels"][model_name] = ranking
                output[item["qid"]]["evidence"][model_name] = channel_evidence
                allowed = allowed_train_qids.get(item["qid"]) if allowed_train_qids else None
                documents: list[str] = []
                seen_documents: set[str] = set()
                for neighbour_rank, neighbour_index in enumerate(memory_indices[row], start=1):
                    neighbour = self.train_questions[int(neighbour_index)]
                    if allowed is not None and neighbour["qid"] not in allowed:
                        continue
                    similarity = max(0.0, float(memory_values[row][neighbour_rank - 1]))
                    for document_id in neighbour["answers"]:
                        if document_id not in seen_documents and len(seen_documents) >= options["query_memory_neighbors"]:
                            continue
                        seen_documents.add(document_id)
                        documents.append(document_id)
                        memory_scores[item["qid"]][document_id] += similarity + 1.0 / (60 + neighbour_rank)
                        memory_models[item["qid"]][document_id].add(model_name)
                        memory_first_rank[item["qid"]].setdefault(document_id, neighbour_rank)
                    if len(seen_documents) >= options["query_memory_neighbors"]:
                        break
            del model
            del vectors, memory_indices
            gc.collect()
        for item in questions:
            qid = item["qid"]
            output[qid]["channels"]["query_memory"] = [
                document_id
                for document_id, _ in sorted(
                    memory_scores[qid].items(),
                    key=lambda pair: (
                        -len(memory_models[qid][pair[0]]),
                        -pair[1],
                        memory_first_rank[qid][pair[0]],
                        pair[0],
                    ),
                )[: options["per_channel_top_k"]]
            ]
            allowed = allowed_train_qids.get(qid) if allowed_train_qids else None
            exact_documents: list[str] = []
            for neighbour in self.train_by_normalized_question[normalize_question(item["question"])]:
                if neighbour["qid"] == qid or (allowed is not None and neighbour["qid"] not in allowed):
                    continue
                exact_documents.extend(neighbour["answers"])
            output[qid]["channels"]["query_exact"] = list(dict.fromkeys(exact_documents))[: options["per_channel_top_k"]]
        return output

    def retrieve(self, question: str, allowed_train_qids: set[str] | None = None) -> dict[str, Any]:
        return self.retrieve_many(
            [{"qid": "single", "question": question}],
            {"single": allowed_train_qids} if allowed_train_qids else None,
        )["single"]

    def fuse(self, retrieval: dict[str, Any], weights: dict[str, float], rrf_k: int) -> dict[str, Any]:
        candidates = rrf(retrieval["channels"], weights, rrf_k, self.config["retrieval"]["fused_top_k"])
        evidence: dict[str, list[str]] = defaultdict(list)
        dense_channels = [name for name, spec in self.config["models"].items() if spec["role"] == "dense"]
        priority = [*dense_channels, "bm25", "accent_char"]
        priority.extend(name for name in retrieval["evidence"] if name not in priority)
        for channel in priority:
            per_doc = retrieval["evidence"].get(channel, {})
            for doc_id in candidates:
                evidence[doc_id].extend(per_doc.get(doc_id, []))
        limit = self.config["reranking"]["evidence_chunks_per_document"]
        return {"candidates": candidates, "evidence": {key: list(dict.fromkeys(value))[:limit] for key, value in evidence.items()}}
