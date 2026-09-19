import json
import tempfile
import unittest
from pathlib import Path

from legalir.prepare import build_corpus
from legalir.storage import read_jsonl


class PrepareTests(unittest.TestCase):
    def test_resume_refreshes_inference_questions_without_rebuilding_corpus(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            contexts = root / "contexts"
            contexts.mkdir()
            (contexts / "context_1.json").write_text(
                json.dumps({"id": "doc-1", "name": "Văn bản", "passage": "Nội dung pháp luật."}),
                encoding="utf-8",
            )
            train_file = root / "train.json"
            public_file = root / "public.json"
            train_file.write_text(
                json.dumps({"train-1": {"question": "Câu huấn luyện?", "answer": ["doc-1"]}}),
                encoding="utf-8",
            )
            public_file.write_text(json.dumps({"public-1": {"question": "Câu public?"}}), encoding="utf-8")
            artifacts = root / "artifacts"
            config = {
                "paths": {
                    "contexts_dir": str(contexts),
                    "train_file": str(train_file),
                    "public_file": str(public_file),
                    "artifacts_dir": str(artifacts),
                },
                "chunking": {
                    "short_words": 32,
                    "short_overlap_words": 4,
                    "long_words": 64,
                    "long_overlap_words": 8,
                },
            }

            first = build_corpus(config)
            corpus_before = (artifacts / "corpus.jsonl").read_bytes()
            public_file.write_text(json.dumps({"private-1": {"question": "Câu private?"}}), encoding="utf-8")
            second = build_corpus(config, resume=True)

            self.assertEqual((artifacts / "corpus.jsonl").read_bytes(), corpus_before)
            self.assertNotEqual(first["public_questions_fingerprint"], second["public_questions_fingerprint"])
            self.assertEqual([row["qid"] for row in read_jsonl(artifacts / "public_questions.jsonl")], ["private-1"])

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
