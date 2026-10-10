# Import Libraries
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import prepare_dataset, AGNewsDataset, LABEL_NAMES

# File paths and directories
part3_dir = Path(__file__).resolve().parent.parent
project_dir = part3_dir.parent
part2_dir = project_dir / "part2"

# Load the trained language model and its vocabulary from Part II.
original_models_dir = part2_dir / "best_models"
checkpoint_path = original_models_dir / "best_lstm_model_pretrained_word2vec_embedding.pt"
vocab_path = original_models_dir / "lstm_pretrained_vocab.json"

# Save the results under Part III.
results_dir = part3_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained LSTM model and its vocabulary.
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
with vocab_path.open("r", encoding="utf-8") as file:
    vocab = json.load(file)

if len(vocab) != state_dict["embedding.weight"].size(0):
    raise ValueError("Vocabulary size does not match the LSTM checkpoint.")

# Prepare the shared AG News splits using the original model vocabulary.
data = prepare_dataset(vocab=vocab)

# One article and one topic label per example; all loaders preserve row order.
train_loader = DataLoader(AGNewsDataset(data["train"], vocab, max_seq_len=64), batch_size=32, shuffle=False)
val_loader = DataLoader(AGNewsDataset(data["validation"], vocab, max_seq_len=64), batch_size=32, shuffle=False)
test_loader = DataLoader(AGNewsDataset(data["test"], vocab, max_seq_len=64), batch_size=32, shuffle=False)


# LSTM Model
class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx):
        super().__init__()

        # Embedding weights are restored from the Part II Word2Vec-pretrained LSTM checkpoint.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def hidden_states(self, token_ids):
        embedded = self.embedding(token_ids)
        output, _ = self.lstm(embedded)
        return output

    def forward(self, token_ids):
        return self.fc(self.hidden_states(token_ids))
    
# Define model hyperparameters (must match the Part II checkpoint).
vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 256
num_layers = 3
pad_idx = vocab["<pad>"]


# Load the trained model, then freeze every parameter for feature extraction.
model = LSTMmodel(vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx=pad_idx)
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.requires_grad_(False)
model.eval()


# Extract frozen LSTM features using mean pooling
@torch.no_grad()
def extract_features(model, data_loader, vocab, device, split_name):
    feature_batches = []
    label_batches = []

    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()

    for token_ids, labels in tqdm(
        data_loader,
        desc=f"Extracting {split_name} features",
    ):
        token_ids = token_ids.to(device, non_blocking=True)

        # Shape: [batch_size, sequence_length, 256]
        hidden = model.hidden_states(token_ids)

        # Pool only article tokens.
        # UNK remains included because it represents an actual text token.
        text_mask = (
            token_ids.ne(vocab["<pad>"])
            & token_ids.ne(vocab["<bos>"])
            & token_ids.ne(vocab["<eos>"])
        )

        hidden = hidden.masked_fill(
            ~text_mask.unsqueeze(-1),
            0.0,
        )

        token_counts = text_mask.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1)

        # Shape: [batch_size, 256]
        pooled = hidden.sum(dim=1) / token_counts

        if not torch.isfinite(pooled).all():
            raise ValueError("Non-finite article features were produced.")

        feature_batches.append(pooled.cpu().numpy())
        label_batches.append(labels.numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()


    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)

    return features, labels


# Extract training and validation features
train_features, train_labels = extract_features(model, train_loader, vocab, device, "training")
val_features, val_labels = extract_features(model, val_loader, vocab, device, "validation")

print(f"Training feature shape: {train_features.shape}")
print(f"Validation feature shape: {val_features.shape}")


# Classifier Training and Validation Loop
c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
classifier_max_iter = 1000
best_score = -1.0
best_c = None
best_classifier = None
validation_results = []

for c in c_values:
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=classifier_max_iter,
            random_state=42,
        ),
    )

    # Both the scaler and logistic regression fit training data only.
    classifier.fit(train_features, train_labels)

    fitted_logreg = classifier.named_steps["logisticregression"]
    if np.max(fitted_logreg.n_iter_) >= classifier_max_iter:
        raise RuntimeError(
            "Logistic regression reached its iteration limit. "
            "Increase classifier_max_iter consistently in both experiments."
        )

    predictions = classifier.predict(val_features)

    accuracy = accuracy_score(val_labels, predictions)
    macro_f1 = f1_score(
        val_labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        average="macro",
        zero_division=0,
    )

    validation_results.append({
        "C": c,
        "validation_accuracy": float(accuracy),
        "validation_macro_f1": float(macro_f1),
    })

    print(
        f"C={c}: validation accuracy={accuracy:.4f}, "
        f"macro-F1={macro_f1:.4f}"
    )

    # On an exact tie, retain the earlier, smaller C.
    if macro_f1 > best_score:
        best_score = macro_f1
        best_c = c
        best_classifier = classifier

classifier = best_classifier


# Evaluate test data only after selecting the classifier.
test_features, test_labels = extract_features(model, test_loader, vocab, device, "test")

predictions = classifier.predict(test_features)

test_accuracy = accuracy_score(test_labels, predictions)
test_macro_f1 = f1_score(
    test_labels,
    predictions,
    labels=list(range(len(LABEL_NAMES))),
    average="macro",
    zero_division=0,
)

print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test macro-F1: {test_macro_f1:.4f}")

report = classification_report(
    test_labels,
    predictions,
    labels=list(range(len(LABEL_NAMES))),
    target_names=list(LABEL_NAMES),
    output_dict=True,
    zero_division=0,
)

experiment_name = "LSTM with Pretrained Word2Vec Embedding on classification of AG News"

metrics = {
    "experiment": experiment_name,
    "best_c": best_c,
    "validation_results": validation_results,
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "classification_report": report,
}

# Save metrics and predictions to files
metrics_path = results_dir / f"lstm_pretrained_metrics.json"
with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

path = results_dir / f"lstm_pretrained_predictions.csv"

with path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "test_index",
        "true_topic",
        "predicted_topic",
        "correct",
        "text",
    ])

    for index, (text, actual, predicted) in enumerate(
        zip(data["test"]["text"], test_labels, predictions)
    ):
        writer.writerow([
            index,
            LABEL_NAMES[int(actual)],
            LABEL_NAMES[int(predicted)],
            int(actual == predicted),
            text,
        ])


matrix = confusion_matrix(
    test_labels,
    predictions,
    labels=list(range(len(LABEL_NAMES))),
)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=list(LABEL_NAMES),
).plot(ax=ax, cmap="Blues", colorbar=False)

ax.set_title("AG News: Frozen LSTM (Pretrained Word2Vec) + Logistic Regression")
fig.tight_layout()
fig.savefig(
    results_dir / "lstm_pretrained_confusion_matrix.png",
    dpi=300,
)
plt.close(fig)

print(f"\nResults saved to: {results_dir}")