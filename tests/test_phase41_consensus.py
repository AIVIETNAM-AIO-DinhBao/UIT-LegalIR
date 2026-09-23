import ast
import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE41 = ROOT / "kaggle" / "phase4_1"
NOTEBOOK = PHASE41 / "legalir_phase41_consensus_cache_only.ipynb"


def load_experiment():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module = types.ModuleType("phase41_consensus_notebook")
    for index in (3, 4):
        exec(compile("".join(notebook["cells"][index]["source"]), f"notebook cell {index}", "exec"),
             module.__dict__)
    return module


class ConsensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = load_experiment()

    def test_fixed_rule_swaps_at_most_once_and_never_uses_labels(self):
        docs = [f"d{i:02d}" for i in range(1, 81)]
        original = docs[:5]
        metadata = [("q1", doc) for doc in docs]
        scores = np.array([0.0] * 80)
        scores[0] = -0.1  # d01 is the unique weakest selected document.
        scores[5] = -0.3  # d06 is within 0.25 of the weakest document.
        scores[6] = -0.4  # d07 is too far below, despite reranker agreement.
        legal = ["d06", "d07"] + [doc for doc in docs if doc not in {"d06", "d07", "d01"}] + ["d01"]
        other = ["d06", "d07"] + [doc for doc in docs if doc not in {"d06", "d07", "d01"}] + ["d01"]
        ranks = {name: {"q1": ranking} for name, ranking in
                 (("legal_reranker", legal), ("jina", other), ("vietnamese_reranker", other))}
        result, changes = self.experiment.consensus_predictions({"q1": original}, scores, metadata, ranks)
        self.assertEqual(set(result["q1"]), set(original[1:] + ["d06"]))
        self.assertEqual(changes, {"eligible_questions": 1, "changed_answer_sets": 1})
        self.assertEqual(len(original), 5)  # Caller-owned Phase 4 baseline is unchanged.
        self.assertEqual(original[0], "d01")

        # Never replace an endorsed weakest document.
        ranks["jina"]["q1"] = ["d01", "d06"] + [doc for doc in docs if doc not in {"d01", "d06"}]
        unchanged, changes = self.experiment.consensus_predictions({"q1": original}, scores, metadata, ranks)
        self.assertEqual(unchanged["q1"], original)
        self.assertEqual(changes["changed_answer_sets"], 0)

    def test_no_swap_without_legal_or_other_agreement_or_near_score(self):
        docs = [f"d{i:02d}" for i in range(1, 81)]
        selected = docs[:5]
        meta = [("q", doc) for doc in docs]
        scores = np.array([-0.1] + [0.] * 4 + [-0.36] + [-1.] * 74)
        legal = ["d06"] + [doc for doc in docs if doc not in {"d06", "d01"}] + ["d01"]
        jina = ["d06"] + [doc for doc in docs if doc not in {"d06", "d01"}] + ["d01"]
        ranks = {name: {"q": ranking} for name, ranking in
                 (("legal_reranker", legal), ("jina", jina), ("vietnamese_reranker", jina))}
        result, _ = self.experiment.consensus_predictions({"q": selected}, scores, meta, ranks)
        self.assertEqual(result["q"], selected)
        scores[5] = -0.3
        ranks["legal_reranker"]["q"] = [doc for doc in docs if doc != "d06"] + ["d06"]
        result, _ = self.experiment.consensus_predictions({"q": selected}, scores, meta, ranks)
        self.assertEqual(result["q"], selected)
        ranks["legal_reranker"]["q"] = legal
        ranks["jina"]["q"] = [doc for doc in docs if doc != "d06"] + ["d06"]
        ranks["vietnamese_reranker"]["q"] = ranks["jina"]["q"]
        result, _ = self.experiment.consensus_predictions({"q": selected}, scores, meta, ranks)
        self.assertEqual(result["q"], selected)

    def test_ranking_mismatch_fails_and_oof_uses_only_other_folds(self):
        meta = [("q", f"d{i}") for i in range(5)]
        with self.assertRaisesRegex(RuntimeError, "ranking/score mismatch"):
            self.experiment.consensus_predictions(
                {"q": [doc for _, doc in meta]}, np.zeros(5), meta,
                {name: {"q": [doc for _, doc in meta[:-1]]} for name in self.experiment.RERANKERS})
        seen = []
        groups = np.repeat(np.arange(5), 2)
        x = np.arange(10, dtype=np.float32).reshape(-1, 1)
        y = np.tile([0, 1], 5)
        metadata = [(f"q{i // 2}", f"d{i}") for i in range(10)]

        class Model:
            def fit(self, features, labels):
                seen.append(set(features[:, 0].astype(int)))
                return self

            def decision_function(self, features):
                return features[:, 0].astype(float)

        with patch.object(self.experiment, "new_model", side_effect=lambda c: Model()) as factory:
            scores = self.experiment.oof_scores(x, y, groups, metadata)
        self.assertEqual(factory.call_count, 5)
        self.assertEqual(factory.call_args.args, (0.1,))
        for fold, training in enumerate(seen):
            self.assertFalse(training.intersection({2 * fold, 2 * fold + 1}))
        np.testing.assert_allclose(scores[::2], -1)
        np.testing.assert_allclose(scores[1::2], 1)

    def test_gate_is_strict_and_requires_each_inner_fold(self):
        old = {"metrics": {"recall": 0.95},
               "per_inner": {str(i): {"recall": 0.9} for i in range(5)}}
        new = {"metrics": {"recall": 0.96},
               "per_inner": {str(i): {"recall": 0.9} for i in range(5)}}
        self.assertTrue(self.experiment.gate(new, old)[0])
        new["metrics"]["recall"] = 0.95
        self.assertFalse(self.experiment.gate(new, old)[0])
        new["metrics"]["recall"] = 0.96
        new["per_inner"]["4"]["recall"] = 0.89
        self.assertFalse(self.experiment.gate(new, old)[0])

    def test_notebook_is_offline_and_runs_logic_in_cells_without_python_files(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 6)
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for source in sources:
            ast.parse(source)
        self.assertIn("CUDA_VISIBLE_DEVICES", sources[0])
        self.assertIn("'--no-index'", sources[1])
        self.assertNotIn("git clone", "\n".join(sources))
        self.assertNotIn("legalir('rerank'", "\n".join(sources))
        self.assertNotIn("phase41_consensus.py", "\n".join(sources))
        self.assertNotIn("phase4_blend.py", "\n".join(sources))
        self.assertIn("def feature_vector(", sources[2])
        self.assertIn("def consensus_predictions(", sources[3])
        self.assertIn("main(PHASE4_OUTPUT, PHASE2_ARTIFACTS, WORK, PRIVATE_FILE)", sources[4])
        for source in sources:
            self.assertNotIn("write_text(", source)
            self.assertNotIn("exec(", source)
            self.assertNotIn("eval(", source)
        self.assertFalse((PHASE41 / "phase41_consensus.py").exists())
        self.assertFalse((PHASE41 / "build_consensus_notebook.py").exists())


if __name__ == "__main__":
    unittest.main()