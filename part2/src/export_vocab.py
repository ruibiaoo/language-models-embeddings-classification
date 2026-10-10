import json
from pathlib import Path
from data import prepare_dataset

# Prepare the original Part II data only; no model training.
_, _, _, vocab, _ = prepare_dataset()

models_dir = Path(__file__).resolve().parent.parent / "best_models"
models_dir.mkdir(parents=True, exist_ok=True)

for filename in [
    "lstm_random_vocab.json",
    "lstm_pretrained_vocab.json",
]:
    output_path = models_dir / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(vocab):,} vocabulary entries to {output_path}")