import random
from collections import Counter

import nltk
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize

# Download the necessary NLTK resources.
def ensure_nltk_resources():
    nltk.download("reuters", quiet=True, raise_on_error=True)
    nltk.download("punkt", quiet=True, raise_on_error=True)
    nltk.download("punkt_tab", quiet=True, raise_on_error=True)
    
# Preprocess the Reuters dataset by lowercasing and tokenizing each article, preserving article boundaries.
def preprocess_reuters(doc_ids):
    return [word_tokenize(reuters.raw(doc_id).lower()) for doc_id in doc_ids]


# Build a vocabulary from the training data, including special tokens for unknown words, padding, beginning of sentence, and end of sentence. Return both the vocabulary and its inverse mapping.
def build_vocabulary(train_data):
    counter = Counter(token for article in train_data for token in article)
    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    inv_vocab = {idx: token for token, idx in vocab.items()}
    return vocab, inv_vocab

# Encode articles by converting tokens to their corresponding IDs using the vocabulary, adding special tokens for beginning and end of sentence.
def encode_articles(data, vocab):
    return [
        [vocab["<bos>"]]
        + [vocab.get(token, vocab["<unk>"]) for token in article]
        + [vocab["<eos>"]]
        for article in data
    ]

# Prepare the Reuters dataset and return training, validation, and test splits along with the vocabulary and inverse vocabulary.
def prepare_reuters():
    ensure_nltk_resources()

    documents = list(reuters.fileids())
    random.Random(42).shuffle(documents)
    train_end = int(0.7 * len(documents))
    val_end = train_end + int(0.15 * len(documents))

    train_docs = documents[:train_end]
    val_docs = documents[train_end:val_end]
    test_docs = documents[val_end:]

    train_data = preprocess_reuters(train_docs)
    val_data = preprocess_reuters(val_docs)
    test_data = preprocess_reuters(test_docs)

    vocab, inv_vocab = build_vocabulary(train_data)
    train_ids = encode_articles(train_data, vocab)
    val_ids = encode_articles(val_data, vocab)
    test_ids = encode_articles(test_data, vocab)

    print(f"Number of documents: {len(documents)}")
    print(f"Training articles:   {len(train_docs)}")
    print(f"Validation articles: {len(val_docs)}")
    print(f"Test articles:       {len(test_docs)}")
    print(f"Vocabulary size: {len(vocab)}")

    return train_ids, val_ids, test_ids, vocab, inv_vocab


if __name__ == "__main__":
    prepare_reuters()
