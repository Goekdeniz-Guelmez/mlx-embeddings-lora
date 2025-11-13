import json
import types
from pathlib import Path
from typing import Any, Dict, List

from datasets import exceptions
from datasets import load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizer


class ContrastiveLearningDataset:
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        anchor_key: str = "anchor",
        positive_key: str = "positive",
        negative_key: str = "negative",
    ):
        self._anchor_data = []
        self._positive_data = []
        self._negative_data = []

        for d in data:
            self._anchor_data.append(tokenizer.encode(d[anchor_key]))
            self._positive_data.append(tokenizer.encode(d[positive_key]))
            if negative_key and negative_key in d:
                self._negative_data.append(
                    tokenizer.encode(d[negative_key], truncation=True)
                )
            else:
                self._negative_data.append(None)

    def __getitem__(self, idx):
        anchor = self._anchor_data[idx]
        positive = self._positive_data[idx]
        negative = (
            self._negative_data[idx]
            if idx < len(self._negative_data) and self._negative_data[idx] is not None
            else None
        )
        return anchor, positive, negative

    def __len__(self):
        return len(self._anchor_data)

    def process(self, d):
        return d


class ConcatenatedDataset:
    def __init__(self, data: List[Any]):
        self._data = data
        self._len = sum(len(d) for d in self._data)

    def __getitem__(self, idx: int):
        for data_idx, data in enumerate(self._data):
            j = idx - len(data)
            if j < 0:
                break
            idx = j
        datum = data[idx]
        datum["_dataset"] = data_idx
        return datum

    def process(self, d):
        return self._data[d["_dataset"]].process(d)

    def __len__(self):
        return self._len


class CacheDataset:
    def __init__(self, data: Any):
        self._data = data
        self._proc_data = [None] * len(data)

    def itemlen(self, idx: int):
        return len(self._data[idx])

    def __getitem__(self, idx: int):
        if self._proc_data[idx] is None:
            self._proc_data[idx] = self._data.process(self._data[idx])
        return self._proc_data[idx]

    def __len__(self):
        return len(self._data)


def create_dataset(
    data,
    tokenizer: PreTrainedTokenizer,
    config,
):
    anchor_key = getattr(config, "anchor_feature", "anchor")
    positive_key = getattr(config, "positive_feature", "positive")
    negative_key = getattr(config, "negative_feature", "negative")

    sample = data[0]

    if (
        anchor_key in sample and positive_key in sample
    ):  # Fixed: was checking if positive_key NOT in sample
        return ContrastiveLearningDataset(
            data, tokenizer, anchor_key, positive_key, negative_key
        )
    else:
        raise ValueError("Unsupported data format for contrastive learning training.")


def load_local_dataset(
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    config,
):
    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(l) for l in fid]
        return create_dataset(data, tokenizer, config)

    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    return train, valid, test


def load_hf_dataset(
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    config,
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        names = ("train", "valid", "test")

        train, valid, test = [
            (
                create_dataset(dataset[n], tokenizer, config)
                if n in dataset.keys()
                else []
            )
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test


def load_custom_hf_dataset(args, tokenizer: PreTrainedTokenizer):
    import datasets

    def create_hf_dataset(dataset_name, config, split, hf_config):
        ds = datasets.load_dataset(
            dataset_name,
            split=split,
            **hf_config,
        )
        return create_dataset(ds, tokenizer, config)

    dataset_collection = args.hf_dataset
    if isinstance(dataset_collection, dict):
        dataset_collection = [dataset_collection]

    collection = []
    for ds in dataset_collection:
        ds_path = ds["path"]
        print(f"Loading Hugging Face dataset {ds_path}.")
        ds["mask_prompt"] = getattr(args, "mask_prompt", False)
        config = types.SimpleNamespace(**ds)
        hf_config = ds.get("config", {})
        if args.train:
            train_split = ds.get("train_split", "train[:80%]")
            valid_split = ds.get("valid_split", "train[-10%:]")
            train = create_hf_dataset(
                ds_path,
                config,
                train_split,
                hf_config,
            )
            valid = create_hf_dataset(
                ds_path,
                config,
                valid_split,
                hf_config,
            )
        else:
            train, valid = [], []

        if args.test:
            test_split = ds.get("test_split")
            test = create_hf_dataset(
                ds_path,
                config,
                test_split,
                hf_config,
            )
        else:
            test = []

        collection.append((train, valid, test))

    if len(collection) == 1:
        return collection[0]

    return tuple(map(ConcatenatedDataset, zip(*collection)))


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if getattr(args, "hf_dataset", False):
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)
        if data_path.exists():
            train, valid, test = load_local_dataset(data_path, tokenizer, args)
        else:
            print(f"Loading Hugging Face dataset {args.data}.")
            train, valid, test = load_hf_dataset(args.data, tokenizer, args)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test
