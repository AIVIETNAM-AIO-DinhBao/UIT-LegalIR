import tempfile
import unittest
from pathlib import Path

from legalir.models import model_load_kwargs, model_source


class ModelLoadingTests(unittest.TestCase):
    def test_hub_model_uses_pinned_revision(self):
        spec = {"id": "example/model", "revision": "a" * 40}
        self.assertEqual(model_source(spec), "example/model")
        self.assertEqual(model_load_kwargs(spec), {"revision": "a" * 40})

    def test_local_snapshot_takes_priority_and_forces_local_loading(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            spec = {
                "id": "example/model",
                "revision": "a" * 40,
                "local_path": temporary_directory,
                "local_files_only": True,
            }
            self.assertEqual(model_source(spec), str(Path(temporary_directory).resolve()))
            self.assertEqual(model_load_kwargs(spec), {"local_files_only": True})

    def test_missing_local_snapshot_is_explicit(self):
        with self.assertRaisesRegex(FileNotFoundError, "local model snapshot"):
            model_source({"id": "example/model", "local_path": "not-a-real-model-directory"})


if __name__ == "__main__":
    unittest.main()