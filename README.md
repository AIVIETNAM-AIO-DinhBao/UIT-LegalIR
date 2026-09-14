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
