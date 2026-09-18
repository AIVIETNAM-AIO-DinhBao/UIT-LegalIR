import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def code_source(notebook: Path) -> str:
    content = json.loads(notebook.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in content["cells"] if cell["cell_type"] == "code")


class OfflineNotebookTests(unittest.TestCase):
    def test_builder_pins_snapshots_without_mutating_kaggle_runtime(self):
        source = code_source(ROOT / "kaggle" / "phase1" / "build_offline_bundle.ipynb")
        self.assertIn("snapshot_download(", source)
        self.assertIn("revision=spec['revision']", source)
        self.assertIn("ignore_patterns=['onnx/*', '*.onnx', '*.onnx_data']", source)
        self.assertIn("'pip', 'download'", source)
        self.assertIn("pip', 'wheel'", source)
        self.assertIn("'--no-deps'", source)
        self.assertNotIn("'pip', 'install'", source)
        self.assertNotIn("from sentence_transformers", source)
        self.assertNotIn("import torch", source)
        self.assertNotIn("'pip', 'check'", source)

        requirements = [
            line
            for line in (ROOT / "requirements-offline.txt").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        self.assertEqual(
            requirements,
            [
                "faiss-cpu>=1.8.0",
                "sentence-transformers==5.7.0",
                "transformers==5.17.0",
                "tokenizers==0.23.2",
                "safetensors==0.8.0",
            ],
        )

    def test_rtx_notebook_has_no_network_install_or_clone(self):
        source = code_source(ROOT / "kaggle" / "phase1" / "legalir_rtx_pro_6000_offline.ipynb")
        self.assertIn("HF_HUB_OFFLINE'] = '1'", source)
        self.assertIn("TRANSFORMERS_OFFLINE'] = '1'", source)
        self.assertIn("'--no-index'", source)
        self.assertIn("'--no-deps'", source)
        self.assertNotIn("'--upgrade'", source)
        self.assertNotIn("'-m', 'venv'", source)
        self.assertIn("'--target', RUNTIME_DIR", source)
        self.assertIn("'--ignore-installed'", source)
        self.assertIn("runtime_env['PYTHONPATH'] = str(RUNTIME_DIR)", source)
        self.assertIn("runtime_env['PYTHONNOUSERSITE'] = '1'", source)
        self.assertIn("'safetensors-0.8.0-'", source)
        self.assertIn("The attached offline bundle is stale", source)
        self.assertIn("run(sys.executable, preflight_path", source)
        self.assertIn("Pinned NLP runtime:", source)
        self.assertIn("selected-contexts' / 'selected-contexts'", source)
        self.assertIn("for model in ('vietlegal_e5', 'vietnamese_embedding', 'nemotron')", source)
        self.assertNotIn("git clone", source)
        self.assertNotIn("huggingface.co", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES': '1'", source)
        self.assertNotIn("'pip', 'check'", source)

    def test_phase3_bundle_and_runtime_are_pinned_and_offline(self):
        directory = ROOT / "kaggle" / "phase3_reranker"
        builder = code_source(directory / "build_reranker_bundle.ipynb")
        runtime = code_source(directory / "legalir_rtx_pro_6000_offline.ipynb")
        config = yaml.safe_load((directory / "kaggle_rtx_pro_6000.yaml").read_text(encoding="utf-8"))

        self.assertIn("phase3-rerankers-harrier-retrieval", builder)
        self.assertIn("kiencnt2205/vietnamese-legal-reranker-bge-base", builder)
        self.assertIn("Qwen/Qwen3-Reranker-0.6B", builder)
        self.assertIn("infgrad/Prism-Qwen3.5-Reranker-0.8B", builder)
        self.assertIn("revision=spec['revision']", builder)
        self.assertNotIn("'pip', 'install'", builder)

        self.assertIn("HF_HUB_OFFLINE", runtime)
        self.assertIn("TRANSFORMERS_OFFLINE", runtime)
        self.assertIn("'--no-index'", runtime)
        self.assertIn("legal_reranker", runtime)
        self.assertIn("qwen3_reranker", runtime)
        self.assertIn("prism_reranker", runtime)
        self.assertNotIn("huggingface.co", runtime)
        self.assertNotIn("git clone", runtime)

        self.assertEqual(config["reranking"]["rerank_top_k"], 20)
        self.assertEqual(config["retrieval"]["calibration_questions"], 400)
        self.assertEqual(
            set(config["models"]),
            {"vietlegal_harrier", "vietnamese_embedding", "nemotron", "legal_reranker", "qwen3_reranker", "prism_reranker"},
        )
        self.assertNotIn("jina", config["models"])
        self.assertNotIn("vietnamese_reranker", config["models"])

    def test_phase2_harrier_notebooks_are_isolated_and_pinned(self):
        directory = ROOT / "kaggle" / "phase2_harrier"
        builder = code_source(directory / "build_offline_bundle.ipynb")
        runtime = code_source(directory / "legalir_rtx_pro_6000_offline.ipynb")
        config = yaml.safe_load((directory / "kaggle_rtx_pro_6000.yaml").read_text(encoding="utf-8"))

        self.assertIn("legalir-phase2-harrier-bundle", builder)
        self.assertIn("mainguyen9/vietlegal-harrier-0.6b", builder)
        self.assertIn("91a0e1ebe4b63b4475bbae40658b8ca9231bea74", builder)
        self.assertIn("revision=spec['revision']", builder)
        self.assertNotIn("'pip', 'install'", builder)

        self.assertIn("HF_HUB_OFFLINE'] = '1'", runtime)
        self.assertIn("legalir-phase2-harrier-run", runtime)
        self.assertIn("phase2-vietlegal-harrier-0.6b", runtime)
        self.assertIn("dense_models =", runtime)
        self.assertNotIn("git clone", runtime)
        self.assertNotIn("huggingface.co", runtime)

        self.assertIn("vietlegal_harrier", config["models"])
        self.assertNotIn("vietlegal_e5", config["models"])
        self.assertEqual(config["models"]["vietlegal_harrier"]["prompt_document"], "")
        self.assertTrue(config["models"]["vietlegal_harrier"]["prompt_query"].startswith("Instruct:"))
        self.assertEqual(config["paths"]["artifacts_dir"], "artifacts_phase2_harrier")


if __name__ == "__main__":
    unittest.main()
