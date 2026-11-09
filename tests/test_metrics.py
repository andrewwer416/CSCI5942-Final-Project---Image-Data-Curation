import unittest

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("PyTorch is not installed") from error

from vlm_train.metrics import (
    image_to_text_ranks,
    recall_at_k,
    retrieval_metrics,
    text_to_image_ranks,
)


# Hand-checked example: three images with two correct captions each.
SIMILARITY = torch.tensor(
    [
        [9.0, 2.0, 8.0, 1.0, 3.0, 0.0],
        [2.0, 3.0, 7.0, 4.0, 9.0, 8.0],
        [1.0, 6.0, 2.0, 3.0, 5.0, 4.0],
    ]
)
CAPTION_TO_IMAGE = torch.tensor([0, 0, 1, 1, 2, 2])


class RetrievalMetricTests(unittest.TestCase):
    def test_image_to_text_uses_best_of_several_correct_captions(self) -> None:
        ranks = image_to_text_ranks(SIMILARITY, CAPTION_TO_IMAGE)
        self.assertEqual(ranks.tolist(), [1, 3, 2])

    def test_text_to_image_ranks_the_single_correct_image(self) -> None:
        ranks = text_to_image_ranks(SIMILARITY, CAPTION_TO_IMAGE)
        self.assertEqual(ranks.tolist(), [1, 3, 2, 1, 2, 2])

    def test_recall_at_k_is_fraction_within_top_k(self) -> None:
        ranks = torch.tensor([1, 3, 2])
        self.assertAlmostEqual(recall_at_k(ranks, 1), 1 / 3)
        self.assertAlmostEqual(recall_at_k(ranks, 2), 2 / 3)
        self.assertAlmostEqual(recall_at_k(ranks, 3), 1.0)

    def test_retrieval_metrics_reports_both_directions(self) -> None:
        metrics = retrieval_metrics(SIMILARITY, CAPTION_TO_IMAGE, ks=(1, 2, 3))
        self.assertAlmostEqual(metrics["i2t_recall@2"], 2 / 3)
        self.assertAlmostEqual(metrics["t2i_recall@1"], 2 / 6)
        self.assertAlmostEqual(metrics["t2i_recall@2"], 5 / 6)
        self.assertAlmostEqual(metrics["t2i_recall@3"], 1.0)

    def test_image_without_a_caption_is_rejected(self) -> None:
        no_captions_for_image_2 = torch.tensor([0, 0, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            image_to_text_ranks(SIMILARITY, no_captions_for_image_2)

    def test_caption_pointing_outside_the_images_is_rejected(self) -> None:
        out_of_range = torch.tensor([0, 0, 1, 1, 2, 7])
        with self.assertRaises(ValueError):
            text_to_image_ranks(SIMILARITY, out_of_range)

    def test_mismatched_caption_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            retrieval_metrics(SIMILARITY, torch.tensor([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
