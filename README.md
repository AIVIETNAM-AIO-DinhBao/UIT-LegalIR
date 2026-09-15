# UIT DSC 2026 – LegalIR

Offline pipeline for Task 1 with three approved dense retrievers, two approved rerankers, lexical retrieval, query-memory and weighted RRF. It performs no remote model inference.

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

1. Run `build_offline_bundle.ipynb` with Internet enabled. It pins and snapshots every model, excludes unused ONNX files, builds the project wheel, downloads the non-Torch dependency wheelhouse, writes checksums, and smoke-tests all models after enabling Hugging Face offline mode.
2. Save its `legalir-offline-bundle/` output as a Kaggle Dataset.
3. Attach that bundle and the competition-data dataset to `legalir_rtx_pro_6000_offline.ipynb`, select RTX Pro 6000, and set Internet to **Off**.

The offline notebook loads the model snapshots by local path, sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before importing Hugging Face libraries, and installs only from the attached wheelhouse with `--no-index`. It intentionally retains Kaggle's installed CUDA-compatible PyTorch build instead of installing a PyPI Torch wheel.

The notebooks intentionally do not run global `pip check`: Kaggle's base image contains unrelated packages with optional version constraints that can conflict with each other. They also install only `faiss-cpu` and `sentence-transformers` without `--upgrade`, preserving Kaggle's tested NumPy, SciPy, scikit-learn, Transformers, and CUDA-enabled PyTorch stack. The offline local-model preflight is the relevant validation because it imports the actual LegalIR stack and runs every model with Hub access disabled.
