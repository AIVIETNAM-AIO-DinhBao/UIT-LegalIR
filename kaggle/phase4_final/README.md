# Phase 4 Final

Deadline-oriented extension of the proven Phase 2 system.

## Kaggle inputs

Attach these four datasets to `legalir_phase4_final.ipynb`:

1. Competition data containing `train.json`, `private-official.json`, and `selected-contexts/selected-contexts/`.
2. `legalir-phase2-harrier-bundle`.
3. `legalir-phase3-reranker-delta`.
4. The saved output of the completed Phase 2 notebook, including the complete `artifacts_phase2_harrier` directory.

Select **RTX Pro 6000**, set Internet to **Off**, and run all cells. The notebook automatically finds the complete Phase 2 artifact directory. If more than one copy is attached, set `PHASE2_ARTIFACTS_OVERRIDE` in the first code cell.

## Output

```text
/kaggle/working/legalir-phase4-private-final/submission_phase4_private_final.zip
```

## What is reused

- Prepared corpus and all dense indexes.
- Train question-memory indexes.
- Phase 2 train/private retrieval caches.
- Phase 2 Jina and Vietnamese reranker top-80 rankings.
- Phase 2 first-stage and final fusion weights as an anchor.

Only the Vietnamese legal BGE reranker is run again, on the existing top-80 candidates. A CPU Logistic Regression blender then uses ranks from all seven retrieval channels, first-stage fusion, all three rerankers, and protected promotion from the Phase 2 top five.