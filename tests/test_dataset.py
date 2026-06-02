import json
import tempfile
import types
import unittest
from pathlib import Path

from mlx_embeddings_lora.trainer.dataset import (
    CacheDataset,
    ConcatenatedDataset,
    ContrastiveLearningDataset,
    load_local_dataset,
)


class FakeTokenizer:
    def encode(self, text, truncation=False):
        return [ord(char) for char in text]


class CountingDataset:
    def __init__(self):
        self.calls = 0
        self.items = [{"text": "alpha"}, {"text": "beta"}]

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)

    def process(self, item):
        self.calls += 1
        return item["text"].upper()


class DatasetTest(unittest.TestCase):
    def test_contrastive_dataset_encodes_optional_negative(self):
        dataset = ContrastiveLearningDataset(
            [
                {"anchor": "a", "positive": "p", "negative": "n"},
                {"anchor": "b", "positive": "q"},
            ],
            FakeTokenizer(),
        )

        self.assertEqual(dataset[0], ([97], [112], [110]))
        self.assertEqual(dataset[1], ([98], [113], None))

    def test_cache_dataset_processes_each_item_once(self):
        source = CountingDataset()
        cached = CacheDataset(source)

        self.assertEqual(cached[0], "ALPHA")
        self.assertEqual(cached[0], "ALPHA")
        self.assertEqual(source.calls, 1)

    def test_concatenated_dataset_dispatches_process_to_source_dataset(self):
        left = CountingDataset()
        right = CountingDataset()
        combined = ConcatenatedDataset([left, right])

        item = combined[2]
        processed = combined.process(item)

        self.assertEqual(processed, "ALPHA")
        self.assertEqual(left.calls, 0)
        self.assertEqual(right.calls, 1)

    def test_load_local_dataset_reads_train_valid_test_jsonl(self):
        config = types.SimpleNamespace(
            anchor_feature="query",
            positive_feature="document",
            negative_feature="negative",
        )

        with tempfile.TemporaryDirectory() as tempdir:
            data_dir = Path(tempdir)
            for split in ("train", "valid", "test"):
                with open(data_dir / f"{split}.jsonl", "w") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "query": f"{split}-q",
                                "document": f"{split}-p",
                                "negative": f"{split}-n",
                            }
                        )
                        + "\n"
                    )

            train, valid, test = load_local_dataset(data_dir, FakeTokenizer(), config)

        self.assertEqual(len(train), 1)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(test), 1)
        self.assertEqual(train[0][0], [ord(char) for char in "train-q"])


if __name__ == "__main__":
    unittest.main()
