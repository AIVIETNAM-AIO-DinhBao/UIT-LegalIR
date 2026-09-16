import unittest

from legalir.fusion import coordinate_search, rrf
from legalir.validation import grouped_folds, score_candidates, score_predictions, validate_submission_shape


class FusionValidationTests(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {"qid": "1", "question": "Điều kiện cấp phép?", "answers": ["10"]},
            {"qid": "2", "question": "Dieu kien cap phep", "answers": ["11", "12"]},
        ]

    def test_rrf_is_deterministic_and_deduplicated(self):
        ranking = rrf({"a": ["10", "11"], "b": ["11", "10", "10"]}, {"a": 1.0, "b": 1.0}, 20, 5)
        self.assertEqual(set(ranking), {"10", "11"})
        self.assertEqual(len(ranking), 2)

    def test_duplicate_questions_stay_in_one_fold(self):
        folds = grouped_folds(self.questions, 5)
        self.assertEqual(folds["1"], folds["2"])

    def test_official_metric_and_submission_constraints(self):
        metrics = score_predictions({"1": ["10"], "2": ["11", "12"]}, self.questions)
        self.assertEqual(metrics, {"recall": 1.0, "precision": 1.0})
        submission = {
            "1": {"answer": ["10", "11", "12", "13", "14"]},
            "2": {"answer": ["11", "12", "10", "13", "14"]},
        }
        validate_submission_shape(submission, self.questions, {"10", "11", "12", "13", "14"})
        submission["1"] = {"answer": ["10"] * 5}
        with self.assertRaises(ValueError):
            validate_submission_shape(submission, self.questions, {"10", "11", "12", "13", "14"})

    def test_candidate_metric_does_not_apply_five_document_submission_limit(self):
        rankings = {"1": ["99", "98", "97", "96", "95", "10"], "2": ["99", "98", "97", "96", "11", "12"]}
        self.assertEqual(score_predictions(rankings, self.questions)["recall"], 0.0)
        self.assertEqual(score_candidates(rankings, self.questions)["candidate_recall"], 1.0)

    def test_coordinate_search_optimizes_candidate_coverage(self):
        rankings = {
            "useful": {"1": ["10"], "2": ["11", "12"]},
            "noise": {"1": ["99"], "2": ["98"]},
        }
        result = coordinate_search(rankings, self.questions, [20], [0.0, 1.0], 20)
        self.assertEqual(result["metrics"]["candidate_recall"], 1.0)
        self.assertEqual(result["weights"]["useful"], 1.0)


if __name__ == "__main__":
    unittest.main()
