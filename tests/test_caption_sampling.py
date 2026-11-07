import unittest

from vlm_data.caption_sampling import caption_index, sample_caption


class CaptionSamplingTests(unittest.TestCase):
    def test_same_inputs_return_same_caption(self) -> None:
        captions = ["a", "b", "c", "d", "e"]
        first = sample_caption(captions, seed=5942, epoch=7, image_id=42)
        second = sample_caption(captions, seed=5942, epoch=7, image_id=42)
        self.assertEqual(first, second)

    def test_index_is_in_range(self) -> None:
        for epoch in range(40):
            value = caption_index(seed=5942, epoch=epoch, image_id=42, caption_count=5)
            self.assertIn(value, range(5))

    def test_rejects_empty_caption_list(self) -> None:
        with self.assertRaises(ValueError):
            sample_caption([], seed=1, epoch=0, image_id=1)


if __name__ == "__main__":
    unittest.main()
