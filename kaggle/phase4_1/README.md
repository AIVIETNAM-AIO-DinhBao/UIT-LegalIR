# Phase 4.1: one cache-only candidate

Run `legalir_phase41_cache_only.ipynb` on Kaggle with Internet **Off** and accelerator **CPU**. Attach the official competition data, the saved Phase 2 **private** notebook output, the saved **fixed Phase 4** notebook output, and the Phase 3 reranker delta bundle (for its offline project wheel). The Phase 4 report in the repository alone is not enough: the saved run must include its prepared text, legal-reranker rankings, config, report, and Phase 4 submission. Phase 2 saved output must include retrieval and Jina/Vietnamese reranker per-engine caches.

The notebook never invokes retrieval/reranking, model downloads, corpus preparation, or GPU. It reconstructs Phase 4's features and five-fold OOF predictions and verifies the saved private submission before creating **one** ensemble of C = (0.03, 0.1, 0.3), equal-weight average of per-question standardized decision scores, alpha = 1, promotion margin = 0.

The only unconditional output is `/kaggle/working/legalir-phase41-private/phase41_report.json`. A Phase 4.1 JSON and ZIP are created in the same directory **only** if OOF Recall exceeds the reproduced Phase 4 OOF, no inner fold declines, and at least one private answer set differs from Phase 4. Otherwise keep Phase 4; do not spend the last submission attempt. Passing this gate does **not** guarantee improvement on private labels.

To regenerate the notebook after modifying its embedded Python files, run `python kaggle/phase4_1/build_notebook.py` from the repository root.