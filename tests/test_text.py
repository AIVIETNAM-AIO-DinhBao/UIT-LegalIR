import unittest

from legalir.text import legal_chunks, normalize_question, strip_accents


class LegalTextTests(unittest.TestCase):
    def test_normalization_groups_accent_and_punctuation_variants(self):
        self.assertEqual(
            normalize_question("Thời hạn cấp, đăng ký xe?") ,
            normalize_question("thoi han cap dang ky xe"),
        )
        self.assertEqual(strip_accents("Điều kiện"), "Dieu kien")

    def test_legal_chunks_preserve_headings_and_split(self):
        text = "Chương I\nQUY ĐỊNH CHUNG\nĐiều 1. Phạm vi\n" + "nội dung " * 60
        short, long = legal_chunks("Luật mẫu", text, 20, 4, 40, 8)
        self.assertGreater(len(short), 1)
        self.assertTrue(all("Luật mẫu" in row["text"] for row in short))
        self.assertTrue(any("Điều 1" in row["heading"] for row in short))
        self.assertLess(len(long), len(short))

    def test_empty_document_still_has_a_retrievable_chunk(self):
        short, long = legal_chunks(None, "", 20, 4, 40, 8)
        self.assertEqual(len(short), 1)
        self.assertEqual(len(long), 1)


if __name__ == "__main__":
    unittest.main()

