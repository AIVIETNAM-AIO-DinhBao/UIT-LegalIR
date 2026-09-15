# UIT DSC 2026 – LegalIR

Offline pipeline for Task 1 with three approved dense retrievers, two approved rerankers, lexical retrieval, query-memory and weighted RRF. It performs no remote model inference.

## System design

The pipeline turns a legal question into exactly five document IDs through the following local-only flow:

```text
competition contexts
  -> article-aware short and long chunks
  -> BM25 + accent-insensitive character retrieval + three dense retrievers
  -> query-memory candidates from similar training questions
  -> tuned weighted RRF first-stage fusion (top 120)
  -> Vietnamese pairwise reranker + Jina listwise reranker
  -> tuned final RRF -> five unique document IDs
```

### Retrieval and reranking models

| Local snapshot | Role in the pipeline | Chunk/input view |
| --- | --- | --- |
| `mainguyen9/vietlegal-e5` | Dense retriever | Short legal chunks; `query:` / `passage:` prompts |
| `AITeamVN/Vietnamese_Embedding_v2` | Dense retriever | Long legal chunks |
| `nvidia/Nemotron-3-Embed-1B-BF16` | Dense retriever | Long legal chunks; `query:` / `passage:` prompts |
| `AITeamVN/Vietnamese_Reranker` | Pairwise reranker | Question-document evidence pairs |
| `jinaai/jina-reranker-v3.5` | Listwise reranker | Candidate windows, then a final top-candidate pass |

Each repository revision is pinned in the Kaggle configuration and model files are loaded from a local path. `audit` counts the exact deployed parameters and rejects a run at or above the four-billion-parameter limit.

### Processing details

- `prepare` preserves document IDs and creates article-aware short (384-token) and long (1,024-token) chunk views.
- The first stage combines word BM25, accent-insensitive character 3–5 gram TF-IDF, the three dense rankings, and a query-memory channel. Query memory transfers document IDs from semantically similar training questions; during training it excludes questions from the held-out fold.
- Candidate chunks are aggregated to document scores using the best chunk plus a small contribution from the second-best chunk. Weighted reciprocal-rank fusion is tuned with five-fold out-of-fold retrieval rankings.
- The two rerankers see local evidence text rather than a remote endpoint. Jina uses round-robin candidate windows to avoid concentrating each window at one end of the first-stage ranking. Final weighted RRF is tuned on the configured validation fold.
- `predict` validates the submission schema and enforces five unique, corpus-valid document IDs for every question.

## Kaggle quick start

Upload the competition data as a Kaggle Dataset, clone this repository, enable Internet only for the initial checkpoint download, then run:

```bash
pip install -e . -r requirements.txt
python -m legalir prepare --config configs/kaggle_t4x2.yaml
python -m legalir audit --config configs/kaggle_t4x2.yaml
python -m legalir index --config configs/kaggle_t4x2.yaml --resume
python -m legalir tune --config configs/kaggle_t4x2.yaml --resume
python -m legalir rerank --config configs/kaggle_t4x2.yaml --split train --fold 0 --resume
python -m legalir tune --config configs/kaggle_t4x2.yaml --final --fold 0 --resume
python -m legalir predict --config configs/kaggle_t4x2.yaml --output submission.json --resume
zip submission.zip submission.json
```

The first `audit` writes `artifacts/model_manifest.json` and blocks a submission if the exact aggregate parameter count is at least four billion. Every long-running stage stores its output under `artifacts/`; use `--resume` when restarting a Kaggle session.

## Commands

- `prepare`: parse selected contexts, preserve document IDs, and create article-aware short/long chunk views.
- `index`: create exact FAISS dense indices, lexical BM25/character indices, and train-question embedding caches.
- `retrieve`: build or resume cached first-stage rankings for `--split train` or `--split public`.
- `tune`: produce 5-fold query-memory OOF retrieval rankings and tune first-stage RRF; `--final` tunes final RRF after reranking a validation fold.
- `rerank`: score a validation fold or the public questions with both local rerankers.
- `predict`: generate a schema-checked `submission.json` with exactly five unique IDs per question.

The task specification ranks by Recall and uses Precision as a tiebreaker; the local pipeline mirrors the scorer's per-question constraints.

## RTX Pro 6000 with Internet disabled

Use the two-notebook workflow under `kaggle/` when the RTX Pro 6000 runtime requires Internet to be disabled:

1. Run `build_offline_bundle.ipynb` with Internet enabled. It pins and snapshots every model, excludes unused ONNX files, builds the project wheel, downloads the small non-Torch wheelhouse, removes Hub metadata caches, writes checksums, and verifies required model and wheel files. It does not install packages or import the runtime stack.
2. Save its `legalir-offline-bundle/` output as a Kaggle Dataset.
3. Attach that bundle and the competition-data dataset to `legalir_rtx_pro_6000_offline.ipynb`, select RTX Pro 6000, and set Internet to **Off**.

The offline notebook loads the model snapshots by local path, sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before importing Hugging Face libraries, and installs only from the attached wheelhouse with `--no-index`. It intentionally retains Kaggle's installed CUDA-compatible PyTorch build instead of installing a PyPI Torch wheel.

The builder never installs packages or imports the model runtime: it only downloads the three small runtime wheels (`faiss-cpu`, `sentence-transformers`, and the Transformers-compatible `tokenizers`), builds the LegalIR wheel, downloads pinned model snapshots, and verifies the resulting files. The offline notebook installs those wheels without `--upgrade`, preserving Kaggle's tested NumPy, SciPy, scikit-learn, Transformers, and CUDA-enabled PyTorch stack. Its local-model preflight is the relevant runtime validation because it imports the actual LegalIR stack and runs every model with Hub access disabled.
