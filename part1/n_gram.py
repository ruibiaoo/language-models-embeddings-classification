import math
import numpy as np

from nltk.lm import KneserNeyInterpolated, Vocabulary
from nltk.util import everygrams
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from time import perf_counter
from data import prepare_reuters
from collections import Counter, defaultdict

# All models use the same shared Reuters preparation.
train_ids, val_ids, test_ids, vocab, inv_vocab = prepare_reuters()

# Class to compute probabilities using Kneser-Ney smoothing efficiently
class FastKN:
    def __init__(self, model):
        self.n = self.order = model.order
        self.D = model.estimator.discount

        self.vocab = model.vocab
        self.counts = model.counts
        self.estimator = model.estimator
        
        n, D = self.n, self.D
        uni = Counter()
        for fd in model.counts[2].values():
            uni.update(w for w, c in fd.items() if c > 0)
        self.uni, self.uni_total = uni, sum(uni.values())
        self.tab = {}                                   
        for L in range(1, n):
            if L == n - 1:                             
                items = ((h, {w: c for w, c in fd.items() if c > 0}) for h, fd in model.counts[L + 1].items())
            else:                                     
                cont = defaultdict(Counter)
                for pre, fd in model.counts[L + 2].items():
                    cont[pre[1:]].update(w for w, c in fd.items() if c > 0)
                items = cont.items()
            raw = model.counts[L + 1]
            for h, num in items:
                total = sum(num.values())
                fd = raw.get(h)
                if not total or not fd: continue
                ntypes = sum(1 for c in fd.values() if c > 0)
                self.tab[h] = (num, total, D * ntypes / total)
 
    def prob(self, w, h):
        w = self.vocab.lookup(w)
        h = tuple(self.vocab.lookup(h[-(self.n - 1):]))
        return self._p(w, h)
 
    def _p(self, w, h):
        if not h:
            return self.uni.get(w, 0) / self.uni_total
        lower = self._p(w, h[1:])
        e = self.tab.get(h)
        if e is None:
            return lower
        num, total, gamma = e
        return max(num.get(w, 0) - self.D, 0.0) / total + gamma * lower
    
    def score(self, word, context=None):
        return self.prob(
            word,
            context if context is not None else ())
    
    def context_counts(self, context):
        if context:
            return self.counts[len(context) + 1][tuple(context)]
        return self.counts.unigrams

# Train n-gram model using Kneser-Ney smoothing
def train_ngram(train_ids, vocab, inv_vocab, n, discount):
    nltk_vocab = Vocabulary([token for token in vocab.keys() if token != "<unk>"], unk_cutoff=1, unk_label="<unk>")
    model = KneserNeyInterpolated(order=n, discount=discount, vocabulary=nltk_vocab)

    for article_ids in train_ids:
        tokens = [inv_vocab[token_id] for token_id in article_ids]
        tokens = ["<bos>"] * (n-2) + tokens
        articles_ngrams = (gram for gram in everygrams(tokens, max_len=n) if gram[-1] != "<bos>")    
        model.fit([articles_ngrams])

    return FastKN(model)


# Train n-gram models for n=2,3,4
n_gram_orders = [2, 3, 4]  
discount = 0.75

models = {}
training_times = {}

for n in n_gram_orders:
    print(f"Training {n}-gram model...")

    training_start = perf_counter()

    model = train_ngram(train_ids=train_ids, vocab=vocab, inv_vocab=inv_vocab, n=n, discount=discount)

    training_seconds = perf_counter() - training_start

    training_times[n] = training_seconds
    models[n] = model

    print(f"Training time for {n}-gram model: {training_seconds:.2f} seconds")


# Function to compute the next token probability using Kneser-Ney smoothing and uniform distribution
def next_token_probability(model, word, history, uniform_weight, vocab_size):
    history = history[-(model.order - 1):]
    kn_probability = model.score(word, history)
    return ((1.0-uniform_weight) * kn_probability) + (uniform_weight * (1.0 / vocab_size))

uniform_weight = 0.001  

# Evaluation 
def evaluate_ngram_model(model, articles_ids, inv_vocab, vocab, uniform_weight, seq_len = 64):
    total_loss = 0.0
    total_tokens = 0
    loss_cache = {}

    for article_ids in tqdm(articles_ids, desc="Evaluating"):
        tokens = [inv_vocab[token_id] for token_id in article_ids]

        for position in range(1, len(tokens)):
            word = tokens[position]

            if word == "<pad>":
                continue

            # Preserve the original 64-token window boundaries.
            window_start = ((position - 1) // seq_len) * seq_len

            # Use only preceding tokens available within this window.
            history_start = max(
                window_start,
                position - (model.order - 1),
            )
            history = tuple(tokens[history_start:position])
            if window_start ==0:
                missing = (model.order - 1) - len(history)
                history = ("<bos>",) * missing + history

            # Each context–word pair has a fixed loss for this model.
            key = (history, word)

            if key not in loss_cache:
                probability = next_token_probability(
                    model=model,
                    word=word,
                    history=history,
                    uniform_weight=uniform_weight,
                    vocab_size=len(vocab),
                )

                loss_cache[key] = -math.log(probability)

            # Every occurrence still contributes to the average.
            total_loss += loss_cache[key]
            total_tokens += 1

    cross_entropy = total_loss / total_tokens
    perplexity = math.exp(cross_entropy)

    return cross_entropy, perplexity


# Evaluate n-gram models on training and validation sets
results = {}
for n in n_gram_orders:
    model = models[n]

    print(f"Evaluating {n}-gram model... on training set")
    train_ce, train_ppl = evaluate_ngram_model(model, train_ids, inv_vocab, vocab, uniform_weight)

    print(f"Evaluating {n}-gram model... on validation set")
    val_ce, val_ppl = evaluate_ngram_model(model, val_ids, inv_vocab, vocab, uniform_weight)

    results[n] = {
        "train_ce": train_ce,
        "train_ppl": train_ppl,
        "val_ce": val_ce,
        "val_ppl": val_ppl
    }
    print(f"{n}-gram training cross-entropy: {train_ce:.4f}")
    print(f"{n}-gram validation cross-entropy: {val_ce:.4f}")
    print(f"{n}-gram validation perplexity: {val_ppl:.2f}")

# Choose the best model based on validation cross-entropy
best_n = min(results, key=lambda n: results[n]["val_ce"])
best_model = models[best_n]
print(f"Best n-gram model: {best_n}-gram")
print(f"Best Validation Cross-Entropy: {results[best_n]['val_ce']:.4f}")


# Count
total_entries = 0

for order in range(1, best_n + 1):
    counts = best_model.counts[order]
    if order == 1:
        entries = sum(count > 0 for count in counts.values())
    else:
        entries = sum(count > 0 for next_word_counts in counts.values() for count in next_word_counts.values())

    total_entries += entries
    print(f"Number of entries in {order}-gram counts: {entries}")
print(f"Total number of entries in {best_n}-gram counts: {total_entries}")


# Evaluate the best model on the test set
print(f"Evaluating best {best_n}-gram model on test set...")

inference_start = perf_counter()

test_ce, test_ppl = evaluate_ngram_model(best_model, test_ids, inv_vocab, vocab, uniform_weight, seq_len= 64)

inference_seconds = perf_counter() - inference_start

print(f"Best {best_n}-gram test cross-entropy: {test_ce:.4f}")
print(f"Best {best_n}-gram test perplexity: {test_ppl:.2f}")
print(f"Inference time: {inference_seconds:.2f} seconds")


# Generate text using the best n-gram model
def kn_distribution(model, history, vocab, cache):
    history = tuple(model.vocab.lookup(history[-(model.order - 1):]))

    if history in cache:
        return cache[history]

    probabilities = np.zeros(len(vocab), dtype=np.float64)

    if not history:
        # Count how many distinct preceding words each word follows.
        counts = Counter()

        for next_counts in model.counts[2].values():
            counts.update(
                word for word, count in next_counts.items()
                if count > 0
            )

        total = sum(counts.values())

        for word, count in counts.items():
            probabilities[vocab[word]] = count / total

    else:
        lower = kn_distribution(model, history[1:], vocab, cache)
        raw_counts = model.context_counts(history)

        if not raw_counts:
            probabilities = lower.copy()

        else:
            if len(history) == model.order - 1:
                counts = raw_counts
            else:
                # Calculate lower-order continuation counts together.
                counts = Counter()

                for prefix, next_counts in model.counts[len(history) + 2].items():
                    if prefix[1:] == history:
                        counts.update(
                            word for word, count in next_counts.items()
                            if count > 0
                        )

            total = sum(counts.values())
            discount = model.estimator.discount

            gamma = (
                discount
                * sum(count > 0 for count in raw_counts.values())
                / total
            )

            probabilities = gamma * lower

            for word, count in counts.items():
                probabilities[vocab[word]] += (
                    max(count - discount, 0) / total
                )

    cache[history] = probabilities
    return probabilities


def generate_ngram_text(
    model, prompt, vocab, inv_vocab, uniform_weight,
    new_tokens=50, context_length=64, temperature=1.0,
    top_k=20, seed=42,
):

    rng = np.random.default_rng(seed)
    cache = {}

    context = ["<bos>"] + [
        token if token in vocab else "<unk>"
        for token in word_tokenize(prompt.lower())
    ]

    generated_words = []

    for _ in range(new_tokens):
        probabilities = kn_distribution(
            model, context[-context_length:], vocab, cache
        )

        # Apply the same uniform mixture used during evaluation.
        probabilities = (
            (1 - uniform_weight) * probabilities
            + uniform_weight / len(vocab)
        )

        # Match RNN/LSTM: generate exactly the requested token count.
        for special in ("<pad>", "<bos>", "<eos>"):
            probabilities[vocab[special]] = 0.0

        # Select the top-k allowed words.
        allowed_ids = np.flatnonzero(probabilities > 0)

        ranked_ids = allowed_ids[
            np.argsort(-probabilities[allowed_ids], kind="stable")
        ]

        top_ids = ranked_ids[:top_k]

        # Apply temperature and normalize the sampling probabilities.
        scores = np.log(probabilities[top_ids]) / temperature
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()

        next_id = int(rng.choice(top_ids, p=weights))
        next_word = inv_vocab[next_id]

        generated_words.append(next_word)
        context.append(next_word)

    return TreebankWordDetokenizer().detokenize(generated_words)

prompts = ["the company", "oil prices", "the bank"]

for prompt in prompts:
    continuation = generate_ngram_text(
        model=best_model,
        prompt=prompt,
        vocab=vocab,
        inv_vocab=inv_vocab,
        uniform_weight=uniform_weight,
        new_tokens=50,
        context_length=64,
        temperature=1.0,
        top_k=20,
        seed=42,
    )
    
    print(f"Prompt: {prompt}\nGenerated continuation: {continuation}\n")
