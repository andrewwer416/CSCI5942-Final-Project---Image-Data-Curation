from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    from PIL import Image
    import torch
except ImportError as error:
    raise unittest.SkipTest("Training dependencies are not installed") from error

from vlm_data.caption_sampling import caption_index
from vlm_data.manifest import Caption, ImageRecord
from vlm_train.data import (
    ASPECT_RATIO_PRESERVING_RGB,
    EvalRetrievalDataset,
    Letterbox,
    TrainRetrievalDataset,
    letterbox_geometry,
    load_aspect_preserving_rgb,
    load_image,
    load_standard_rgb,
    padding_patch_mask,
)
from vlm_train.model import ModelConfig, RetrievalModel, contrastive_loss
from vlm_train.tokenizer import Vocabulary


def tiny_model_config(vocabulary: Vocabulary) -> ModelConfig:
    return ModelConfig(
        vocab_size=len(vocabulary),
        pad_id=vocabulary.pad_id,
        eos_id=vocabulary.eos_id,
        image_size=32,
        image_dim=24,
        image_layers=1,
        image_heads=2,
        image_feedforward=48,
        max_length=8,
        text_dim=24,
        text_layers=1,
        text_heads=2,
        text_feedforward=48,
        embed_dim=16,
    )


class VocabularyTests(unittest.TestCase):
    def test_vocabulary_is_ordered_by_frequency_then_word(self) -> None:
        vocabulary = Vocabulary.from_texts(["b a", "a c", "c a"], min_frequency=2)
        self.assertEqual(vocabulary.words[4:], ["a", "c"])

    def test_encode_adds_markers_maps_unknowns_and_pads(self) -> None:
        vocabulary = Vocabulary.from_texts(["a dog"], min_frequency=1)
        ids = vocabulary.encode("A cat", max_length=5)
        self.assertEqual(
            ids,
            [
                vocabulary.bos_id,
                vocabulary.ids["a"],
                vocabulary.unk_id,
                vocabulary.eos_id,
                vocabulary.pad_id,
            ],
        )

    def test_long_captions_keep_eos_within_max_length(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], min_frequency=1)
        ids = vocabulary.encode("a a a a a a", max_length=4)
        self.assertEqual(len(ids), 4)
        self.assertEqual(ids[-1], vocabulary.eos_id)


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "train2017").mkdir()
        pixels = np.zeros((10, 20, 3), dtype=np.uint8)
        pixels[..., 0] = 255
        Image.fromarray(pixels).save(self.root / "train2017" / "a.png")
        texts = ["zero", "one", "two", "three", "four"]
        self.record = ImageRecord(
            image_id=7,
            split="train2017",
            file_name="a.png",
            relative_path="train2017/a.png",
            width=20,
            height=10,
            captions=tuple(Caption(index, text) for index, text in enumerate(texts)),
            category_ids=(),
            object_count=0,
        )
        self.texts = texts
        self.vocabulary = Vocabulary.from_texts(texts, min_frequency=1)

    def test_standard_rgb_is_square_and_scaled(self) -> None:
        image = load_standard_rgb(self.root / "train2017" / "a.png", 16)
        self.assertEqual(image.shape, (3, 16, 16))
        self.assertAlmostEqual(image[0].mean().item(), 1.0)
        self.assertAlmostEqual(image[1].mean().item(), -1.0)

    def test_training_caption_follows_the_deterministic_sampler(self) -> None:
        dataset = TrainRetrievalDataset(
            [self.record],
            image_root=self.root,
            vocabulary=self.vocabulary,
            seed=5942,
            image_size=16,
            max_length=4,
        )
        for epoch in range(10):
            dataset.set_epoch(epoch)
            expected = self.texts[
                caption_index(seed=5942, epoch=epoch, image_id=7, caption_count=5)
            ]
            _image, _mask, tokens = dataset[0]
            self.assertEqual(tokens.tolist(), self.vocabulary.encode(expected, 4))

    def test_eval_dataset_returns_every_caption(self) -> None:
        dataset = EvalRetrievalDataset(
            [self.record],
            image_root=self.root,
            vocabulary=self.vocabulary,
            image_size=16,
            max_length=4,
        )
        _image, _mask, tokens = dataset[0]
        self.assertEqual(tokens.shape, (5, 4))


class LetterboxGeometryTests(unittest.TestCase):
    def test_landscape_portrait_and_square_fill_the_long_side_centered(self) -> None:
        self.assertEqual(letterbox_geometry(640, 480, 640), Letterbox(0, 80, 640, 480))
        self.assertEqual(letterbox_geometry(480, 640, 640), Letterbox(80, 0, 480, 640))
        self.assertEqual(letterbox_geometry(500, 500, 640), Letterbox(0, 0, 640, 640))

    def test_small_images_are_upscaled_like_p1(self) -> None:
        self.assertEqual(letterbox_geometry(320, 240, 640), Letterbox(0, 80, 640, 480))

    def test_short_side_rounds_to_nearest_pixel(self) -> None:
        # 333 x 500 scales the width to 426.24 -> 426.
        self.assertEqual(letterbox_geometry(333, 500, 640).width, 426)
        # 3:2 at 1000 x 667 scales to 426.88 -> 427.
        self.assertEqual(letterbox_geometry(1000, 667, 640).height, 427)

    def test_uneven_padding_puts_the_extra_pixel_bottom_or_right(self) -> None:
        box = letterbox_geometry(640, 479, 640)
        self.assertEqual((box.top, 640 - box.top - box.height), (80, 81))
        box = letterbox_geometry(479, 640, 640)
        self.assertEqual((box.left, 640 - box.left - box.width), (80, 81))

    def test_extreme_ratios_keep_at_least_one_pixel(self) -> None:
        self.assertEqual(letterbox_geometry(1000, 100, 640), Letterbox(0, 288, 640, 64))
        self.assertEqual(letterbox_geometry(5000, 1, 640).height, 1)

    def test_invalid_sizes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            letterbox_geometry(0, 10, 640)


class PaddingMaskTests(unittest.TestCase):
    def test_four_by_three_masks_five_patch_rows_top_and_bottom(self) -> None:
        mask = padding_patch_mask(letterbox_geometry(640, 480, 640), 640, 16).view(40, 40)
        self.assertEqual(int(mask.sum()), 400)
        self.assertTrue(mask[:5].all() and mask[35:].all())
        self.assertFalse(mask[5:35].any())

    def test_partially_padded_border_patches_stay_visible(self) -> None:
        # Content spans rows 81-558: patch rows 5 (80-95) and 34 (544-559) are partial.
        box = letterbox_geometry(640, 478, 640)
        self.assertEqual(box.top, 81)
        mask = padding_patch_mask(box, 640, 16).view(40, 40)
        self.assertTrue(mask[:5].all() and mask[35:].all())
        self.assertFalse(mask[5].any() or mask[34].any())

    def test_portrait_masks_columns_and_square_masks_nothing(self) -> None:
        mask = padding_patch_mask(letterbox_geometry(480, 640, 640), 640, 16).view(40, 40)
        self.assertTrue(mask[:, :5].all() and mask[:, 35:].all())
        self.assertFalse(mask[:, 5:35].any())
        self.assertFalse(padding_patch_mask(letterbox_geometry(9, 9, 640), 640, 16).any())

    def test_image_size_must_be_a_patch_multiple(self) -> None:
        with self.assertRaises(ValueError):
            padding_patch_mask(Letterbox(0, 0, 10, 10), 10, 4)


class AspectPreservingLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "red.png"
        pixels = np.zeros((10, 20, 3), dtype=np.uint8)
        pixels[..., 0] = 255
        Image.fromarray(pixels).save(self.path)

    def test_content_is_centered_on_mid_gray_padding(self) -> None:
        image, mask = load_aspect_preserving_rgb(self.path, 16, 4)
        self.assertEqual(image.shape, (3, 16, 16))
        # 20 x 10 -> 16 x 8 content at rows 4-11.
        self.assertTrue(torch.equal(image[:, :4], torch.zeros(3, 4, 16)))
        self.assertTrue(torch.equal(image[:, 12:], torch.zeros(3, 4, 16)))
        self.assertTrue(torch.allclose(image[0, 4:12], torch.ones(8, 16)))
        self.assertTrue(torch.allclose(image[1:, 4:12], -torch.ones(2, 8, 16)))
        expected = torch.tensor([True] * 4 + [False] * 8 + [True] * 4)
        self.assertTrue(torch.equal(mask, expected))

    def test_standard_rgb_returns_an_empty_mask(self) -> None:
        image, mask = load_image(
            self.path, preprocessing="standard_rgb", image_size=16, patch_size=4
        )
        self.assertTrue(torch.equal(image, load_standard_rgb(self.path, 16)))
        self.assertEqual(mask.shape, (16,))
        self.assertFalse(mask.any())

    def test_unknown_preprocessing_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_image(self.path, preprocessing="nope", image_size=16, patch_size=4)

    def test_datasets_use_the_selected_preprocessing(self) -> None:
        record = ImageRecord(
            image_id=7,
            split="train2017",
            file_name="red.png",
            relative_path="red.png",
            width=20,
            height=10,
            captions=tuple(Caption(index, f"c{index}") for index in range(5)),
            category_ids=(),
            object_count=0,
        )
        vocabulary = Vocabulary.from_texts(["c0"], min_frequency=1)
        common = dict(
            image_root=self.path.parent,
            vocabulary=vocabulary,
            image_size=16,
            max_length=4,
            preprocessing=ASPECT_RATIO_PRESERVING_RGB,
            patch_size=4,
        )
        expected_image, expected_mask = load_aspect_preserving_rgb(self.path, 16, 4)
        for dataset in (
            TrainRetrievalDataset([record], seed=5942, **common),
            EvalRetrievalDataset([record], **common),
        ):
            image, mask, _tokens = dataset[0]
            self.assertTrue(torch.equal(image, expected_image))
            self.assertTrue(torch.equal(mask, expected_mask))


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5942)
        self.vocabulary = Vocabulary.from_texts(["a red dog", "a blue cat"], min_frequency=1)
        self.model = RetrievalModel(tiny_model_config(self.vocabulary))
        self.images = torch.randn(2, 3, 32, 32)
        self.tokens = torch.tensor(
            [
                self.vocabulary.encode("a red dog", 8),
                self.vocabulary.encode("a blue cat", 8),
            ]
        )

    def test_logits_compare_every_image_with_every_caption(self) -> None:
        self.assertEqual(self.model(self.images, self.tokens).shape, (2, 2))

    def test_embeddings_are_unit_length(self) -> None:
        norms = self.model.encode_text(self.tokens).norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(2)))

    def test_padding_does_not_change_the_text_embedding(self) -> None:
        self.model.eval()
        short = torch.tensor([self.vocabulary.encode("a red dog", 6)])
        long = torch.tensor([self.vocabulary.encode("a red dog", 8)])
        self.assertTrue(
            torch.allclose(self.model.encode_text(short), self.model.encode_text(long), atol=1e-5)
        )

    def test_masked_patches_cannot_change_the_image_embedding(self) -> None:
        self.model.eval()
        # 32 px / 16 px patches = 2 x 2; mask the bottom row of patches.
        mask = torch.tensor([[False, False, True, True]] * 2)
        changed = self.images.clone()
        changed[:, :, 16:] = torch.randn(2, 3, 16, 32)
        before = self.model.encode_image(self.images, mask)
        after = self.model.encode_image(changed, mask)
        self.assertTrue(torch.allclose(before, after, atol=1e-5))
        self.assertFalse(torch.allclose(before, self.model.encode_image(changed), atol=1e-3))

    def test_empty_mask_matches_no_mask(self) -> None:
        self.model.eval()
        empty = torch.zeros(2, 4, dtype=torch.bool)
        self.assertTrue(
            torch.allclose(
                self.model.encode_image(self.images, empty),
                self.model.encode_image(self.images),
                atol=1e-6,
            )
        )

    def test_mask_shape_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            self.model.encode_image(self.images, torch.zeros(2, 3, dtype=torch.bool))

    def test_correct_pairs_give_a_lower_loss(self) -> None:
        matched = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        swapped = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        self.assertLess(contrastive_loss(matched), contrastive_loss(swapped))

    def test_training_steps_lower_the_loss_on_one_batch(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        first_loss = contrastive_loss(self.model(self.images, self.tokens)).item()
        for _ in range(30):
            optimizer.zero_grad()
            contrastive_loss(self.model(self.images, self.tokens)).backward()
            optimizer.step()
        final_loss = contrastive_loss(self.model(self.images, self.tokens)).item()
        self.assertLess(final_loss, first_loss)


if __name__ == "__main__":
    unittest.main()
