from collections import Counter
from pathlib import Path
from datasets import DatasetDict, load_dataset
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset

import nltk
import numpy as np
import torch


DATASET_NAME = "fancyzhx/ag_news"
TRAIN_EXAMPLES = 50000
VAL_EXAMPLES = 10000
SEED = 42
LABEL_NAMES = ("World", "Sports", "Business", "Sci/Tech")


# Download the necessary NLTK resources.
def ensure_nltk_resources():
    download_dir = Path(__file__).resolve().parents[1] / ".cache" / "nltk_data"
    download_dir.mkdir(parents=True, exist_ok=True)
    download_path = str(download_dir.resolve())
    if download_path not in nltk.data.path:
        nltk.data.path.insert(0, download_path)
    nltk.download(
        "punkt_tab", download_dir=download_path,
        quiet=True, raise_on_error=True,
    )


def normalize_text(text):
    return " ".join(text.split()).lower()


def check_dataset(dataset):
    # Reserve test texts first, so matching training articles are excluded.
    test_texts = {
        normalize_text(text)
        for text in dataset["test"]["text"]
    }

    seen = set()
    kept_indices = []
    duplicate_count = 0
    overlap_count = 0
    empty_count = 0

    for index, example in enumerate(dataset["train"]):
        key = normalize_text(example["text"])

        if not key:
            empty_count += 1
            continue

        if key in test_texts:
            overlap_count += 1
            continue

        if key in seen:
            duplicate_count += 1
            continue

        seen.add(key)
        kept_indices.append(index)

    cleaned_train = dataset["train"].select(kept_indices)

    return DatasetDict({
        "train": cleaned_train,
        "test": dataset["test"],
    })

def split_dataset(dataset, train_size=TRAIN_EXAMPLES, val_size=VAL_EXAMPLES,
                  seed=SEED):
    num_classes = len(LABEL_NAMES)

    labels = np.asarray(dataset["train"]["label"])


    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    train_per_class = train_size // num_classes
    val_per_class = val_size // num_classes

    for label in range(num_classes):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)

        train_indices.extend(indices[:train_per_class].tolist())
        val_indices.extend(
            indices[train_per_class:train_per_class + val_per_class].tolist()
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return DatasetDict({
        "train": dataset["train"].select(train_indices),
        "validation": dataset["train"].select(val_indices),
        "test": dataset["test"],
    })


def tokenize_batch(batch):
    return {"tokens": [
        word_tokenize(normalize_text(text))
        for text in batch["text"]
    ]}


def encode_batch(batch, vocab):
    return {"input_ids": [
        [vocab["<bos>"]]
        + [vocab.get(token, vocab["<unk>"]) for token in tokens]
        + [vocab["<eos>"]]
        for tokens in batch["tokens"]
    ]}

class AGNewsDataset(Dataset):
    def __init__(self, split, vocab, max_seq_len=64):
        self.articles = split["input_ids"]
        self.labels = split["label"]

        self.max_seq_len = max_seq_len
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.pad_id = vocab["<pad>"]

        if max_seq_len < 3:
            raise ValueError("Need space for BOS, at least one token, and EOS.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text_ids = self.articles[index][1:-1]
        text_ids = text_ids[:self.max_seq_len - 2]

        input_ids = [self.bos_id] + text_ids + [self.eos_id]

       
        padding_length = self.max_seq_len - len(input_ids)
        input_ids += [self.pad_id] * padding_length

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )

def prepare_dataset(vocab=None, train_size=TRAIN_EXAMPLES,
                    val_size=VAL_EXAMPLES, seed=SEED):

    ensure_nltk_resources()

    if vocab is not None:
        required = {"<unk>", "<pad>", "<bos>", "<eos>"}
        if not required.issubset(vocab):
            raise ValueError("The LM vocabulary must include all four special tokens.")

    raw = load_dataset(DATASET_NAME)
    cleaned = check_dataset(raw)
    data = split_dataset(cleaned, train_size, val_size, seed)

    data = data.map(tokenize_batch, batched=True, desc="Tokenizing AG News")

    if vocab is not None:
        data = data.map(
            encode_batch, batched=True, fn_kwargs={"vocab": vocab},
            desc="Encoding with the trained language-model vocabulary",
        )

    print(f"Dataset: {DATASET_NAME}")
    print(f"Split seed: {seed}")

    for name, split in data.items():
        counts = Counter(split["label"])
        print(f"{name.capitalize()} examples: {len(split):,}")
        print("  " + ", ".join(
            f"{label_name}: {counts[label]:,}"
            for label, label_name in enumerate(LABEL_NAMES)
        ))

        # add this block — unconditional, doesn't depend on vocab
        over_limit = sum(len(tokens) > 64 - 2 for tokens in split["tokens"])
        print(f"  Truncated at {64}-token limit: {over_limit}/{len(split):,} "
              f"({over_limit/len(split):.2%})")

        if vocab is not None:
            token_count = sum(len(tokens) for tokens in split["tokens"])
            unknown_count = sum(
                token not in vocab for tokens in split["tokens"] for token in tokens
            )
            rate = unknown_count / token_count if token_count else 0.0
            print(f"Text-token OOV rate: {rate:.2%}")

    return data


if __name__ == "__main__":
    prepare_dataset()
