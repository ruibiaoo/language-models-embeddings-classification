from collections import Counter
from hashlib import sha256
from pathlib import Path

import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from ftfy import fix_text

DATASET_NAME = "roneneldan/TinyStories"
TRAIN_STORIES = 20000
VAL_STORIES = 2000
TEST_STORIES = 2000
SEED = 42
SHUFFLE_BUFFER_SIZE = 10000

# Download the necessary NLTK resources.
def ensure_nltk_resources():
    # Use an explicit project cache instead of the Windows default data path.
    download_dir = Path(__file__).resolve().parents[1] / ".cache" / "nltk_data"
    download_dir.mkdir(parents=True, exist_ok=True)
    download_path = str(download_dir.resolve())
    if download_path not in nltk.data.path:
        nltk.data.path.insert(0, download_path)
    nltk.download(
        "punkt_tab", download_dir=download_path,
        quiet=True, raise_on_error=True,
    )


# Load the Tinystories dataset
def load_stories(split, count, seen):
    stream = load_dataset(DATASET_NAME, split=split, streaming=True)
    stream = stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

    print(f"Loading {count:,} unique stories from TinyStories {split}...")
    data = []
    for example in stream:
            text = fix_text(example["text"], unescape_html=False)
            text = text.replace("\u00e2\u20ac", " ")
            cleaned = " ".join(text.split()).lower()

            if not cleaned:
                continue

            # One shared set prevents identical normalized stories appearing
            # more than once, including across training/validation/test.
            fingerprint = sha256(cleaned.encode("utf-8")).digest()
            if fingerprint in seen:
                continue

            tokens = word_tokenize(cleaned)
            if not tokens:
                continue

            seen.add(fingerprint)
            data.append(tokens)
            if len(data) == count:
                return data

    raise ValueError(
        f"Requested {count:,} unique stories from {split}, "
        f"but only found {len(data):,}."
    )

# Keep every distinct training token, plus the four special tokens.
def build_vocabulary(train_data):
    counter = Counter(token for story in train_data for token in story)
    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    for token, count in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    inv_vocab = {idx: token for token, idx in vocab.items()}
    return vocab, inv_vocab

# Keep each story intact, with beginning/end-of-story markers.
def encode_stories(data, vocab):
    return [
        [vocab["<bos>"]]
        + [vocab.get(token, vocab["<unk>"]) for token in story]
        + [vocab["<eos>"]]
        for story in data
    ]

# Prepare the dataset and return training, validation, and test splits along with the vocabulary and inverse vocabulary.
def prepare_dataset():
    ensure_nltk_resources()
    seen = set()
    train_data = load_stories("train", TRAIN_STORIES, seen)

    # TinyStories provides no test split. Partition a separate held-out
    # selection from its validation split into our validation and test sets.
    held_out = load_stories("validation", VAL_STORIES + TEST_STORIES, seen)
    val_data = held_out[:VAL_STORIES]
    test_data = held_out[VAL_STORIES:]

    vocab, inv_vocab = build_vocabulary(train_data)
    train_ids = encode_stories(train_data, vocab)
    val_ids = encode_stories(val_data, vocab)
    test_ids = encode_stories(test_data, vocab)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Buffered shuffle seed: {SEED}")
    print(f"Training stories:   {len(train_data):,}")
    print(f"Validation stories: {len(val_data):,}")
    print(f"Test stories:       {len(test_data):,}")
    print(f"Vocabulary size: {len(vocab)}")

    for name, split_ids in (
        ("Training", train_ids),
        ("Validation", val_ids),
        ("Test", test_ids),
    ):
        # Exclude BOS/EOS from the text-token counts and unknown rates.
        total = sum(len(ids) - 2 for ids in split_ids)
        unknown = sum(ids.count(vocab["<unk>"]) for ids in split_ids)
        print(f"{name} text tokens: {total:,}; <unk> rate: {unknown / total:.2%}")

    return train_ids, val_ids, test_ids, vocab, inv_vocab


if __name__ == "__main__":
    prepare_dataset()
