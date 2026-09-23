import ast
import importlib.util
import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE41 = ROOT / "kaggle" / "phase4_1"
PHASE4 = ROOT / "kaggle" / "phase4_final"


def load_experiment():
    sys.path.insert(0, str(PHASE4))
    try:
        spec = importlib.util.spec_from_file_location("phase41_ensemble", PHASE41 / "phase41_ensemble.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(PHASE4))


class Phase41Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = load_experiment()

    def test_notebook_embeds_actual_source_and_never_runs_expensive_jobs(self):
        notebook = json.loads((PHASE41 / "legalir_phase41_cache_only.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len(notebook["cells"]), 4)
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for source in sources:
            ast.parse(source)
        self.assertIn("CUDA_VISIBLE_DEVICES", sources[0])
        self.assertIn("'--no-index'", sources[1])
        self.assertNotIn("git clone", "\n".join(sources))
        self.assertNotIn("legalir('rerank'", "\n".join(sources))
        tree = ast.parse(sources[-1])
        for filename, expected in (("phase4_blend.py", PHASE4 / "phase4_blend.py"),
                                   ("phase41_ensemble.py", PHASE41 / "phase41_ensemble.py")):
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"
                     and isinstance(node.func.value, ast.BinOp) and isinstance(node.func.value.right, ast.Constant)
                     and node.func.value.right.value == filename]
            self.assertEqual(len(calls), 1)
            self.assertEqual(ast.literal_eval(calls[0].args[0]), expected.read_text(encoding="utf-8"))

    def test_gate_requires_strict_gain_and_no_inner_regression(self):
        old = {"metrics": {"recall": 0.95},
               "per_inner": {str(i): {"recall": 0.9} for i in range(5)}}
        new = {"metrics": {"recall": 0.96},
               "per_inner": {str(i): {"recall": 0.91} for i in range(5)}}
        self.assertTrue(self.experiment.gate(new, old)[0])
        new["metrics"]["recall"] = 0.95
        self.assertFalse(self.experiment.gate(new, old)[0])
        new["metrics"]["recall"] = 0.96
        new["per_inner"]["3"]["recall"] = 0.89
        self.assertFalse(self.experiment.gate(new, old)[0])

    def test_oof_trains_only_on_other_question_groups_and_normalizes_each_model(self):
        # Probe without fitting: a record of training masks validates grouping.
        seen = []
        groups = np.repeat(np.arange(5), 2)
        x = np.arange(10, dtype=np.float32).reshape(-1, 1)
        y = np.tile([0, 1], 5)
        meta = [(f"q{i // 2}", f"d{i}") for i in range(10)]

        class Model:
            def fit(self, features, labels):
                seen.append(set(features[:, 0].astype(int)))
                return self

            def decision_function(self, features):
                return features[:, 0].astype(float)

        with patch.object(self.experiment, "new_model", return_value=Model()):
            result = self.experiment.oof_scores(x, y, groups, meta)
        self.assertEqual(set(result), set(self.experiment.CS))
        self.assertEqual(len(seen), 15)
        for i, train_set in enumerate(seen):
            self.assertFalse(train_set.intersection({2 * (i % 5), 2 * (i % 5) + 1}))
        for scores in result.values():
            np.testing.assert_allclose(scores[::2], -1)
            np.testing.assert_allclose(scores[1::2], 1)

    def test_cache_guard_and_phase4_private_reconstruction_guard(self):
        with self.assertRaisesRegex(FileNotFoundError, "Required cache"):
            self.experiment.required(PHASE41 / "not-a-cache.json")
        with self.assertRaisesRegex(RuntimeError, "QIDs differ"):
            self.experiment.compare_phase4({"q1": ["a"]}, {"q2": {"answer": ["a"]}})
        self.assertEqual(self.experiment.compare_phase4({"q1": ["a", "b"]},
                                                           {"q1": {"answer": ["b", "a"]}}), 0)

    def test_verify_source_rejects_mismatched_private_input_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "private-official.json"
            private.write_text('{"q1":{"question":"test"}}', encoding="utf-8")
            report = {"experiment_id": "phase4-phase2-legal-bge-protected-blender",
                      "test_file": "private-official.json", "test_questions": 1,
                      "test_sha256": "not-the-private-input-hash"}
            with self.assertRaisesRegex(RuntimeError, "report does not match"):
                self.experiment.verify_source(report, private, root, root)
            report["test_sha256"] = hashlib.sha256(private.read_bytes()).hexdigest()
            with self.assertRaisesRegex(FileNotFoundError, "prepare_manifest"):
                self.experiment.verify_source(report, private, root, root)


if __name__ == "__main__":
    unittest.main()