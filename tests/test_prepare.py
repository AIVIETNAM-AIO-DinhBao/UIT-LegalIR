import json
import tempfile
import unittest
from pathlib import Path

from legalir.prepare import build_corpus


class PrepareTests(unittest.TestCase):
    def test_empty_context_directory_fails_before_writing_empty_corpus(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            contexts = root / "contexts"
            contexts.mkdir()
            train_file = root / "train.json"
            public_file = root / "public.json"
            train_file.write_text(json.dumps({}), encoding="utf-8")
            public_file.write_text(json.dumps({}), encoding="utf-8")
            config = {
                "paths": {
                    "contexts_dir": str(contexts),
                    "train_file": str(train_file),
                    "public_file": str(public_file),
                    "artifacts_dir": str(root / "artifacts"),
                },
                "chunking": {"short_tokens": 32, "short_overlap": 4, "long_tokens": 64, "long_overlap": 8},
            }

            with self.assertRaisesRegex(FileNotFoundError, r"context_\*\.json"):
                build_corpus(config)

            self.assertFalse((root / "artifacts" / "corpus.jsonl").exists())


if __name__ == "__main__":
    unittest.main()