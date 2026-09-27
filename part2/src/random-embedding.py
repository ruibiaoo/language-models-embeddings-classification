# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import re

from torch.utils.data import DataLoader, Dataset as TorchDataset
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from time import perf_counter
from pathlib import Path
from data import prepare_dataset

# Resolve output locations relative to this file, not the working directory.
part1_dir = Path(__file__).resolve().parent.parent
results_dir = part1_dir / "results"
models_dir = part1_dir / "best_models"
results_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = models_dir / "best_lstm_model_random_embedding.pt"

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All models use the same shared dataset preparation.
train_ids, val_ids, test_ids, vocab, inv_vocab = prepare_dataset()


# Dataset class for loading tokenized paragraphs
class Dataset(TorchDataset):
    def __init__(self, articles, seq_len=64, stride=1):

        self.articles = articles
        self.seq_len = seq_len
        self.pad_id = vocab["<pad>"]

        # Store positions instead of copying overlapping token sequences.
        # Short articles and final partial windows are kept and padded.
        self.windows = [
            (article_idx, start)
            for article_idx, article in enumerate(articles)
            for start in range(0, len(article) - 1, stride)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        article_idx, start = self.windows[idx]
        article = self.articles[article_idx]
        end = min(start + self.seq_len, len(article) - 1)

        x = article[start:end]
        y = article[start + 1:end + 1]
        padding = [self.pad_id] * (self.seq_len - len(x))
        return (
            torch.tensor(x + padding, dtype=torch.long),
            torch.tensor(y + padding, dtype=torch.long),
        )


# All splits use 64-token windows with stride 64
train_loader = DataLoader(Dataset(train_ids, seq_len = 64, stride=64), batch_size=32,shuffle=True)
val_loader = DataLoader(Dataset(val_ids, seq_len = 64, stride = 64), batch_size=32, shuffle=False)
test_loader = DataLoader(Dataset(test_ids, seq_len = 64, stride= 64), batch_size=32, shuffle=False)


# LSTM Model
class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx):
        super().__init__()

        # Randomly initialized, trainable embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        output, _ = self.lstm(embedded)
        return self.fc(output)

# Define model hyperparameters
vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 256
num_layers = 3
pad_idx = vocab["<pad>"]

# Initialize the model
torch.manual_seed(42)
model = LSTMmodel(vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx=pad_idx).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Count the number of trainable parameters
num_parameters = sum(p.numel() for p in model.parameters())
print (f"Number of trainable parameters: {num_parameters}")


# Training and Validation Loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_epoch = None
num_epochs = 50

training_seconds = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_tokens = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_training_start_time = perf_counter()

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()

        n_tokens = (y != pad_idx).sum().item()
        train_loss += loss.item() * n_tokens
        train_tokens += n_tokens

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds += perf_counter() - epoch_training_start_time

    avg_loss = train_loss / train_tokens
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
    print(f"Total training time: {training_seconds:.2f} seconds")

    # Validation
    model.eval()
    val_loss = 0.0
    val_tokens = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))

            n_tokens = (y != pad_idx).sum().item()
            val_loss += loss.item() * n_tokens
            val_tokens += n_tokens

    avg_val_loss = val_loss / val_tokens
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), checkpoint_path)
        
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Training loss")
plt.plot(epochs, val_losses, label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Average next-token cross-entropy")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(results_dir / "lstm_random_300_loss_curve.png", dpi=300)
plt.show()


# Evaluate the best-validation checkpoint on the test set.
test_loss = 0.0
test_tokens = 0
test_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
model.eval()

if device.type == "cuda":
    torch.cuda.synchronize()
inference_start = perf_counter()

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating on test set"):
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = test_criterion(output.reshape(-1, output.size(-1)),y.reshape(-1))

        test_loss += loss.item()
        test_tokens += (y != pad_idx).sum().item()

if device.type == "cuda":
    torch.cuda.synchronize()
inference_seconds = perf_counter() - inference_start

print(f"Total inference time: {inference_seconds:.2f} seconds")

test_ce = test_loss / test_tokens
test_ppl = math.exp(test_ce)

print(f"Test cross-entropy: {test_ce:.4f}")
print(f"Test perplexity: {test_ppl:.2f}")


# Generate text using the trained model
@torch.no_grad()
def generate_text(model, prompt, vocab, inv_vocab, 
                  new_tokens=50, context_length=64,
                temperature=1.0, seed = 42):
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    prompt_tokens = word_tokenize(prompt.lower())
    context = [vocab["<bos>"]] + [vocab.get(token, vocab["<unk>"])for token in prompt_tokens]

    generated_ids = []

    for _ in range(new_tokens):
        inputs = torch.tensor(
            [context[-context_length:]],
            dtype=torch.long,
            device=device,
        )

        # Use the prediction after the last input token.
        scores = model(inputs)[0, -1].clone()

        # Generate exactly new_tokens ordinary/unknown tokens.
        for special in ["<pad>", "<bos>", "<eos>", "<unk>"]:
            scores[vocab[special]] = float("-inf")

        top_scores, top_ids = torch.topk(scores, k = 20)

        probabilities = torch.softmax(
            top_scores / temperature, dim=-1
        )

        selected = torch.multinomial(
            probabilities, 1, generator=generator
        )
        next_id = top_ids[selected].item()

        generated_ids.append(next_id)
        context.append(next_id)

    words = [inv_vocab[token_id] for token_id in generated_ids]

    text = TreebankWordDetokenizer().detokenize(words)
    return re.sub(r"\s+\.", ".", text)


# Text Generation
prompts = ["elinor said", "marianne was", "she could not"]

for prompt in prompts:
    continuation = generate_text(model = model,
                                 prompt = prompt,
                                 vocab = vocab,
                                 inv_vocab = inv_vocab,
                                 new_tokens = 50,
                                 context_length = 64,
                                 temperature = 1.0,
                                 seed = 42)
    print(prompt + " " + continuation)
