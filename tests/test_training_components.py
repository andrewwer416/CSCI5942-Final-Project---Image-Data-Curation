from dataclasses import replace
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
import math
import random

from vlm_train.data import (
    ASPECT_RATIO_PRESERVING_RGB,
    RGB_PLUS_CLIP_FEATURES,
    ClipFeatureTable,
    COLOR_CONTRAST_NORMALIZATION,
    RANDOM_RESIZED_CROP,
    CropBox,
    CropKey,
    EvalRetrievalDataset,
    Letterbox,
    TrainRetrievalDataset,
    letterbox_geometry,
    normalize_appearance,
    load_aspect_preserving_rgb,
    load_color_contrast_normalized,
    load_image,
    load_random_resized_crop,
    load_standard_rgb,
    padding_patch_mask,
    sample_crop_box,
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
            _image, _mask, _clip, tokens = dataset[0]
            self.assertEqual(tokens.tolist(), self.vocabulary.encode(expected, 4))

    def test_eval_dataset_returns_every_caption(self) -> None:
        dataset = EvalRetrievalDataset(
            [self.record],
            image_root=self.root,
            vocabulary=self.vocabulary,
            image_size=16,
            max_length=4,
        )
        _image, _mask, _clip, tokens = dataset[0]
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
            image, mask, _clip, _tokens = dataset[0]
            self.assertTrue(torch.equal(image, expected_image))
            self.assertTrue(torch.equal(mask, expected_mask))


class CropSamplingTests(unittest.TestCase):
    def boxes(self, width: int, height: int, count: int = 2000) -> list[CropBox]:
        return [
            sample_crop_box(width, height, seed=5942, epoch=epoch, image_id=7)
            for epoch in range(count)
        ]

    def test_same_key_gives_the_same_crop_regardless_of_global_rng(self) -> None:
        first = sample_crop_box(640, 480, seed=5942, epoch=3, image_id=11)
        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        self.assertEqual(sample_crop_box(640, 480, seed=5942, epoch=3, image_id=11), first)

    def test_crops_change_with_epoch_image_and_seed(self) -> None:
        self.assertGreater(len(set(self.boxes(640, 480, 20))), 15)
        base = sample_crop_box(640, 480, seed=5942, epoch=0, image_id=7)
        self.assertNotEqual(sample_crop_box(640, 480, seed=5942, epoch=0, image_id=8), base)
        self.assertNotEqual(sample_crop_box(640, 480, seed=5943, epoch=0, image_id=7), base)

    def test_crops_stay_inside_the_image_and_within_the_ranges(self) -> None:
        for width, height in [(640, 480), (480, 640), (640, 427), (500, 500), (640, 213)]:
            image_ratio = width / height
            for box in self.boxes(width, height, 300):
                self.assertGreaterEqual(box.left, 0)
                self.assertGreaterEqual(box.top, 0)
                self.assertLessEqual(box.left + box.width, width)
                self.assertLessEqual(box.top + box.height, height)
                area = box.width * box.height / (width * height)
                self.assertGreaterEqual(area, 0.5 - 0.01)
                jitter = (box.width / box.height) / image_ratio
                self.assertTrue(0.8 - 0.02 <= jitter <= 1.25 + 0.02, (width, height, box))

    def test_area_is_uniform_and_ratio_is_log_uniform(self) -> None:
        boxes = self.boxes(640, 480)
        areas = [box.width * box.height / (640 * 480) for box in boxes]
        log_jitter = [math.log((box.width / box.height) / (640 / 480)) for box in boxes]
        # A crop near full area only fits when its shape stays near the image's, so such
        # attempts are redrawn: accepted crops average ~0.70 of the area, not the 0.75 drawn.
        self.assertAlmostEqual(sum(areas) / len(areas), 0.70, delta=0.02)
        self.assertAlmostEqual(sum(log_jitter) / len(log_jitter), 0.0, delta=0.02)

    def test_no_fitting_attempt_falls_back_to_the_full_image(self) -> None:
        box = sample_crop_box(640, 480, seed=5942, epoch=0, image_id=7, attempts=0)
        self.assertEqual(box, CropBox(0, 0, 640, 480))

    def test_invalid_ranges_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_crop_box(640, 480, seed=1, epoch=0, image_id=1, min_scale=0)
        with self.assertRaises(ValueError):
            sample_crop_box(0, 480, seed=1, epoch=0, image_id=1)


class RandomResizedCropLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Left half red, right half blue, so a crop's position shows in its colors.
        pixels = np.zeros((30, 40, 3), dtype=np.uint8)
        pixels[:, :20, 0] = 255
        pixels[:, 20:, 2] = 255
        self.path = self.root / "halves.png"
        Image.fromarray(pixels).save(self.path)
        self.record = ImageRecord(
            image_id=7,
            split="train2017",
            file_name="halves.png",
            relative_path="halves.png",
            width=40,
            height=30,
            captions=tuple(Caption(index, f"c{index}") for index in range(5)),
            category_ids=(),
            object_count=0,
        )
        self.vocabulary = Vocabulary.from_texts(["c0"], min_frequency=1)

    def test_crop_is_taken_from_stored_pixels_then_stretched_to_a_square(self) -> None:
        key = dict(seed=5942, epoch=0, image_id=7)
        box = sample_crop_box(40, 30, **key)
        image = load_random_resized_crop(self.path, 16, **key)
        self.assertEqual(image.shape, (3, 16, 16))
        red_columns = max(0, min(box.left + box.width, 20) - box.left)
        expected_red = red_columns / box.width
        observed_red = ((image[0] + 1) / 2).mean().item()
        self.assertAlmostEqual(observed_red, expected_red, delta=0.08)

    def test_training_uses_the_epoch_crop_and_evaluation_uses_p1(self) -> None:
        common = dict(
            image_root=self.root,
            vocabulary=self.vocabulary,
            image_size=16,
            max_length=4,
            preprocessing=RANDOM_RESIZED_CROP,
            patch_size=4,
        )
        train = TrainRetrievalDataset([self.record], seed=5942, **common)
        for epoch in (0, 5):
            train.set_epoch(epoch)
            image, mask, _clip, _tokens = train[0]
            expected = load_random_resized_crop(self.path, 16, seed=5942, epoch=epoch, image_id=7)
            self.assertTrue(torch.equal(image, expected))
            self.assertFalse(mask.any())
        image, mask, _clip, _tokens = EvalRetrievalDataset([self.record], **common)[0]
        self.assertTrue(torch.equal(image, load_standard_rgb(self.path, 16)))
        self.assertFalse(mask.any())

    def test_crop_without_a_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_image(self.path, preprocessing=RANDOM_RESIZED_CROP, image_size=16, patch_size=4)
        image, _mask = load_image(
            self.path,
            preprocessing=RANDOM_RESIZED_CROP,
            image_size=16,
            patch_size=4,
            crop_key=CropKey(seed=5942, epoch=1, image_id=7),
        )
        expected = load_random_resized_crop(self.path, 16, seed=5942, epoch=1, image_id=7)
        self.assertTrue(torch.equal(image, expected))


def _luma(pixels: np.ndarray) -> np.ndarray:
    return pixels.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


class AppearanceNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(5942)

    def test_color_cast_is_removed(self) -> None:
        texture = self.rng.uniform(60, 200, size=(48, 64, 1))
        cast = np.rint(texture * np.array([0.7, 1.0, 1.0])).astype(np.uint8)
        output = normalize_appearance(cast).astype(np.float64) / 255
        norms = np.mean(output.reshape(-1, 3) ** 6, axis=0) ** (1 / 6)
        self.assertLess(norms.max() - norms.min(), 0.02)
        self.assertLess(norms[0] / norms[1], 1.02)  # red was 30% weaker before

    def test_white_balance_gain_is_capped_at_two(self) -> None:
        # Uncapped, blue would be multiplied by about 7.
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[..., :2] = self.rng.integers(180, 220, size=(32, 32, 1))
        pixels[..., 2] = self.rng.integers(20, 40, size=(32, 32))
        output = normalize_appearance(pixels).astype(np.float64)
        source = pixels.astype(np.float64) / 255
        norms = np.mean(source.reshape(-1, 3) ** 6, axis=0) ** (1 / 6)
        gains = norms.mean() / norms
        self.assertGreater(gains[2], 4)  # the cap binds
        capped = np.clip(gains, 0.5, 2.0)
        # The contrast step shifts all channels equally, so the green-blue gap is exactly
        # what capped white balance leaves; uncapped, it would be about zero.
        expected_gap = (source[..., 1] * capped[1] - source[..., 2] * capped[2]).mean() * 255
        self.assertAlmostEqual((output[..., 1] - output[..., 2]).mean(), expected_gap, delta=1)
        self.assertGreater(expected_gap, 60)

    def test_grayscale_stays_gray_and_low_contrast_doubles_at_most(self) -> None:
        gray = self.rng.integers(100, 140, size=(40, 40)).astype(np.uint8)
        output = normalize_appearance(np.repeat(gray[..., None], 3, axis=2))
        self.assertTrue(np.array_equal(output[..., 0], output[..., 1]))
        self.assertTrue(np.array_equal(output[..., 1], output[..., 2]))
        source_spread = np.percentile(gray, 99) - np.percentile(gray, 1)
        output_spread = np.percentile(output[..., 0], 99) - np.percentile(output[..., 0], 1)
        self.assertAlmostEqual(output_spread / source_spread, 2.0, delta=0.05)
        # The stretch is centered on the percentile midpoint, which maps to mid-gray.
        midpoint = (np.percentile(gray, 1) + np.percentile(gray, 99)) / 2
        nearest = np.abs(gray.astype(np.float64) - midpoint).argmin()
        self.assertAlmostEqual(
            float(output[..., 0].flat[nearest]),
            127.5 + 2 * (gray.flat[nearest] - midpoint),
            delta=1,
        )

    def test_contrast_maps_percentiles_to_full_range(self) -> None:
        values = self.rng.uniform(51, 204, size=(50, 50))
        pixels = np.repeat(np.rint(values)[..., None], 3, axis=2).astype(np.uint8)
        luma = _luma(normalize_appearance(pixels))
        self.assertAlmostEqual(np.percentile(luma, 1), 0, delta=2)
        self.assertAlmostEqual(np.percentile(luma, 99), 255, delta=2)

    def test_contrast_step_preserves_chroma(self) -> None:
        # Shuffles of the same values give every channel the same p-norm, so white
        # balance is the identity and only the luma stretch acts.
        values = self.rng.uniform(0.42, 0.58, size=40 * 40) * 255
        pixels = np.stack([self.rng.permutation(values) for _ in range(3)], axis=1)
        pixels = np.rint(pixels).astype(np.uint8).reshape(40, 40, 3)
        output = normalize_appearance(pixels).astype(np.float64)
        source = pixels.astype(np.float64)
        for channel in (0, 2):  # Cr ~ R - Y and Cb ~ B - Y
            before = source[..., channel] - _luma(pixels)
            after = output[..., channel] - _luma(output.astype(np.uint8))
            self.assertLess(np.abs(after - before).max(), 1.5)
        self.assertGreater(_luma(output.astype(np.uint8)).std(), 1.5 * _luma(pixels).std())

    def test_flat_and_invalid_inputs(self) -> None:
        flat = np.full((8, 8, 3), 30, dtype=np.uint8)
        self.assertTrue(np.all(normalize_appearance(flat) == 128))
        black = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertTrue(np.all(normalize_appearance(black) == 128))
        with self.assertRaises(ValueError):
            normalize_appearance(np.zeros((8, 8), dtype=np.uint8))
        with self.assertRaises(ValueError):
            normalize_appearance(np.zeros((8, 8, 3), dtype=np.float32))

    def test_training_and_evaluation_both_use_the_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.rint(self.rng.uniform(60, 200, size=(30, 40, 1)) * [0.7, 1.0, 1.0])
            Image.fromarray(pixels.astype(np.uint8)).save(root / "cast.png")
            expected = Image.fromarray(normalize_appearance(pixels.astype(np.uint8)))
            expected = expected.resize((16, 16), Image.Resampling.BILINEAR)
            expected_tensor = torch.from_numpy(np.array(expected)).permute(2, 0, 1) / 127.5 - 1
            loaded = load_color_contrast_normalized(root / "cast.png", 16)
            self.assertTrue(torch.allclose(loaded, expected_tensor.float()))
            record = ImageRecord(
                image_id=3,
                split="train2017",
                file_name="cast.png",
                relative_path="cast.png",
                width=40,
                height=30,
                captions=tuple(Caption(index, f"c{index}") for index in range(5)),
                category_ids=(),
                object_count=0,
            )
            common = dict(
                image_root=root,
                vocabulary=Vocabulary.from_texts(["c0"], min_frequency=1),
                image_size=16,
                max_length=4,
                preprocessing=COLOR_CONTRAST_NORMALIZATION,
                patch_size=4,
            )
            for dataset in (
                TrainRetrievalDataset([record], seed=5942, **common),
                EvalRetrievalDataset([record], **common),
            ):
                image, mask, _clip, _tokens = dataset[0]
                self.assertTrue(torch.equal(image, loaded))
                self.assertFalse(mask.any())


class ClipFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5942)
        vectors = rng.normal(size=(6, 4)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self.ids = np.array([3, 5, 8, 13, 21, 34])
        self.table = ClipFeatureTable(self.ids, vectors)

    def test_lookup_by_image_id(self) -> None:
        self.assertTrue(torch.equal(self.table[13], torch.from_numpy(self.table.vectors[3])))
        with self.assertRaises(KeyError):
            self.table[4]

    def test_derangement_moves_every_vector_and_is_reproducible(self) -> None:
        shuffled = self.table.deranged(self.ids.tolist(), seed=5942)
        moved = [not torch.equal(shuffled[int(i)], self.table[int(i)]) for i in self.ids]
        self.assertTrue(all(moved))
        originals = {tuple(self.table[int(i)].tolist()) for i in self.ids}
        self.assertEqual({tuple(shuffled[int(i)].tolist()) for i in self.ids}, originals)
        again = self.table.deranged(self.ids.tolist(), seed=5942)
        self.assertTrue(all(torch.equal(again[int(i)], shuffled[int(i)]) for i in self.ids))

    def test_unsorted_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ClipFeatureTable(np.array([2, 1]), np.zeros((2, 4), dtype=np.float32))

    def test_p5_datasets_return_p1_images_with_the_clip_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.zeros((10, 20, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / "a.png")
            record = ImageRecord(
                image_id=13,
                split="train2017",
                file_name="a.png",
                relative_path="a.png",
                width=20,
                height=10,
                captions=tuple(Caption(index, f"c{index}") for index in range(5)),
                category_ids=(),
                object_count=0,
            )
            common = dict(
                image_root=root,
                vocabulary=Vocabulary.from_texts(["c0"], min_frequency=1),
                image_size=16,
                max_length=4,
                preprocessing=RGB_PLUS_CLIP_FEATURES,
                patch_size=4,
                clip_features=self.table,
            )
            for dataset in (
                TrainRetrievalDataset([record], seed=5942, **common),
                EvalRetrievalDataset([record], **common),
            ):
                image, mask, clip, _tokens = dataset[0]
                self.assertTrue(torch.equal(image, load_standard_rgb(root / "a.png", 16)))
                self.assertFalse(mask.any())
                self.assertTrue(torch.equal(clip, self.table[13]))
            with self.assertRaises(ValueError):
                TrainRetrievalDataset([record], seed=5942, **{**common, "clip_features": None})
            with self.assertRaises(ValueError):
                EvalRetrievalDataset([record], **{**common, "preprocessing": "standard_rgb"})


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

    def test_clip_fusion_leaves_other_weights_identical_to_p1(self) -> None:
        config = tiny_model_config(self.vocabulary)
        torch.manual_seed(7)
        plain = RetrievalModel(config)
        torch.manual_seed(7)
        fused = RetrievalModel(replace(config, clip_dim=4))
        fused_state = fused.state_dict()
        for name, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, fused_state[name]), name)
        self.assertEqual(set(fused_state) - set(plain.state_dict()), {"clip_projection.weight"})
        self.assertEqual(tuple(fused_state["clip_projection.weight"].shape), (24, 4))
        torch.manual_seed(7)
        again = RetrievalModel(replace(config, clip_dim=4))
        self.assertTrue(torch.equal(again.clip_projection.weight, fused.clip_projection.weight))

    def test_zero_clip_vector_gives_the_encoder_only_embedding(self) -> None:
        config = tiny_model_config(self.vocabulary)
        torch.manual_seed(7)
        plain = RetrievalModel(config).eval()
        torch.manual_seed(7)
        fused = RetrievalModel(replace(config, clip_dim=4)).eval()
        zero = torch.zeros(2, 4)
        self.assertTrue(
            torch.allclose(
                fused.encode_image(self.images, None, zero),
                plain.encode_image(self.images),
                atol=1e-6,
            )
        )
        real = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
        self.assertFalse(
            torch.allclose(
                fused.encode_image(self.images, None, real), plain.encode_image(self.images)
            )
        )

    def test_clip_features_are_required_exactly_when_fused(self) -> None:
        config = tiny_model_config(self.vocabulary)
        fused = RetrievalModel(replace(config, clip_dim=4))
        with self.assertRaises(ValueError):
            fused.encode_image(self.images)
        with self.assertRaises(ValueError):
            self.model.encode_image(self.images, None, torch.zeros(2, 4))

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
