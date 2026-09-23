import ast
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "kaggle" / "phase4_final"


def load_blender():
    spec = importlib.util.spec_from_file_location("phase4_blend", PHASE4 / "phase4_blend.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase4FinalTests(unittest.TestCase):
    def test_notebook_is_self_contained_offline_and_syntactically_valid(self):
        notebook = json.loads((PHASE4 / "legalir_phase4_final.ipynb").read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        self.assertIn("HF_HUB_OFFLINE", source)
        self.assertIn("PHASE2_ARTIFACTS_OVERRIDE", source)
        self.assertIn("rerank_top_k'] = 80", source)
        self.assertIn("pairwise_max_length'] = 512", source)
        self.assertIn("legal_reranker", source)
        self.assertIn("phase4_blend.py", source)
        self.assertNotIn("git clone", source)
        self.assertNotIn("huggingface.co", source)

    def test_protected_promotion_preserves_five_unique_documents(self):
        blender = load_blender()
        metadata = [("q1", document) for document in ("a", "b", "c", "d", "e", "f")]
        scores = np.asarray([1.0, 0.9, 0.8, 0.7, 0.1, 0.95])
        prediction, audit = blender.protected_predictions(
            scores,
            metadata,
            {"q1": ["a", "b", "c", "d", "e"]},
            margin=0.25,
        )
        self.assertEqual(set(prediction["q1"]), {"a", "b", "c", "d", "f"})
        self.assertEqual(len(prediction["q1"]), 5)
        self.assertEqual(audit, {"promoted_questions": 1, "promotions": 1})

    def test_inner_fold_is_deterministic_and_not_the_outer_hash(self):
        blender = load_blender()
        questions = [f"Câu hỏi pháp luật số {index}" for index in range(100)]
        assignments = [blender.inner_fold(question) for question in questions]
        self.assertEqual(assignments, [blender.inner_fold(question) for question in questions])
        self.assertEqual(set(assignments), set(range(5)))


if __name__ == "__main__":
    unittest.main()