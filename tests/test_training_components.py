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
from vlm_train.data import EvalRetrievalDataset, TrainRetrievalDataset, load_standard_rgb
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
            _image, tokens = dataset[0]
            self.assertEqual(tokens.tolist(), self.vocabulary.encode(expected, 4))

    def test_eval_dataset_returns_every_caption(self) -> None:
        dataset = EvalRetrievalDataset(
            [self.record],
            image_root=self.root,
            vocabulary=self.vocabulary,
            image_size=16,
            max_length=4,
        )
        _image, tokens = dataset[0]
        self.assertEqual(tokens.shape, (5, 4))


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
