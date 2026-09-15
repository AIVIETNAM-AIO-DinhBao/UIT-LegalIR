import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def code_source(notebook: Path) -> str:
    content = json.loads(notebook.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in content["cells"] if cell["cell_type"] == "code")


class OfflineNotebookTests(unittest.TestCase):
    def test_builder_pins_snapshots_and_runs_offline_smoke_tests(self):
        source = code_source(ROOT / "kaggle" / "build_offline_bundle.ipynb")
        self.assertIn("snapshot_download(", source)
        self.assertIn("revision=spec['revision']", source)
        self.assertIn("ignore_patterns=['onnx/*', '*.onnx', '*.onnx_data']", source)
        self.assertIn("HF_HUB_OFFLINE'] = '1'", source)
        self.assertIn("JinaListwiseReranker", source)
        self.assertIn("pip', 'wheel'", source)
        self.assertIn("'--no-deps'", source)
        self.assertIn("'hf-xet'", source)
        self.assertNotIn("'pip', 'check'", source)

        requirements = (ROOT / "requirements-offline.txt").read_text(encoding="utf-8")
        self.assertIn("hf-xet>=1.1.3,<2.0.0", requirements)

    def test_rtx_notebook_has_no_network_install_or_clone(self):
        source = code_source(ROOT / "kaggle" / "legalir_rtx_pro_6000_offline.ipynb")
        self.assertIn("HF_HUB_OFFLINE'] = '1'", source)
        self.assertIn("TRANSFORMERS_OFFLINE'] = '1'", source)
        self.assertIn("'--no-index'", source)
        self.assertIn("'--no-deps'", source)
        self.assertIn("selected-contexts' / 'selected-contexts'", source)
        self.assertIn("for model in ('vietlegal_e5', 'vietnamese_embedding', 'nemotron')", source)
        self.assertNotIn("git clone", source)
        self.assertNotIn("huggingface.co", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES': '1'", source)
        self.assertNotIn("'pip', 'check'", source)


if __name__ == "__main__":
    unittest.main()
