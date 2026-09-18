# Phase 3 strong reranker workflow

This workflow keeps the Phase 2 Harrier retrieval stack and replaces both Phase 2
rerankers with three local rerankers:

- `kiencnt2205/vietnamese-legal-reranker-bge-base`
- `Qwen/Qwen3-Reranker-0.6B`
- `infgrad/Prism-Qwen3.5-Reranker-0.8B`

The exact deployed parameter total is `3,930,936,897`, below the four-billion
limit. The runtime reranks only the first-stage top 20 and calibrates final RRF
on 400 deterministic training questions.

## Run order

1. Run `build_reranker_bundle.ipynb` with Internet enabled.
2. Create a Kaggle Dataset from `legalir-phase3-reranker-delta`.
3. Attach the competition-data Dataset, the existing Phase 2 Harrier bundle
   Dataset, and the new Phase 3 delta Dataset to
   `legalir_rtx_pro_6000_offline.ipynb`.
4. Replace the three `REPLACE_WITH_*_SLUG` values in the first runtime cell.
5. Select RTX Pro 6000 and set Internet to Off.
6. Run all cells. The final upload is
   `/kaggle/working/legalir-phase3-run/submission_phase3_tuned.zip`.

The Phase 2 bundle supplies only the three dense retrievers used by Phase 3;
the old Jina and Vietnamese reranker snapshots are not configured or loaded.