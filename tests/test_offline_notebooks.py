import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def code_source(notebook: Path) -> str:
    content = json.loads(notebook.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in content["cells"] if cell["cell_type"] == "code")


class OfflineNotebookTests(unittest.TestCase):
    def test_builder_pins_snapshots_without_mutating_kaggle_runtime(self):
        source = code_source(ROOT / "kaggle" / "build_offline_bundle.ipynb")
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
        source = code_source(ROOT / "kaggle" / "legalir_rtx_pro_6000_offline.ipynb")
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


if __name__ == "__main__":
    unittest.main()
