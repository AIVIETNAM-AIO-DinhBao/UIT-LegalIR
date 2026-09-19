# Private-test runbook

Both runtime notebooks use `private-official.json` by default. The pipeline
continues to call the inference split `public` internally; the generated config
maps that split to the selected test file.

## Required Kaggle inputs

- Competition data containing `train.json`, `private-official.json`, and
  `selected-contexts/selected-contexts/`.
- The rebuilt Phase 2 Harrier bundle.
- For Phase 3, the rebuilt Phase 3 reranker delta bundle.
- Optionally, an artifacts directory saved from a completed run of the same
  phase. Set `CHECKPOINT_ARTIFACTS_DIR` to that directory to reuse corpus,
  indexes, train question memory, calibrated weights, and completed train
  reranking caches.

A Phase 2 checkpoint may also seed Phase 3's shared retrieval assets. Phase 3
will detect that the reranker model set differs and will not import Phase 2
final weights or reranker caches.

## Build order

1. Commit and push the source and notebook fixes. Copy that commit SHA into the
   `REPO_REF` cell of both bundle builders instead of leaving it as `main`.
2. Run `phase2_harrier/build_offline_bundle.ipynb` with Internet enabled and
   publish its output as a Kaggle Dataset.
3. Run `phase3_reranker/build_reranker_bundle.ipynb` with Internet enabled and
   publish its output as a Kaggle Dataset.
4. Attach the required inputs, replace every `REPLACE_WITH_*_SLUG`, and leave
   `TEST_FILENAME = 'private-official.json'`.

## Cache safety

Each notebook records the test filename, question count, and SHA-256. When the
input changes it removes only test-dependent retrieval and reranking caches.
Corpus indexes and train calibration artifacts remain reusable. A checkpoint
from a Kaggle input Dataset is read through symlinks for large immutable files;
the notebook never modifies that input Dataset.

Expected private outputs:

- Phase 2: `submission_phase2_private_harrier.zip`
- Phase 3: `submission_phase3_private_tuned.zip`

The notebook validates that every question receives exactly five unique corpus
document IDs before writing the submission.
