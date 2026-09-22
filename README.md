# DSA4213 Assignment

## Part 1: Next-token language model comparison

Part 1 compares an interpolated Kneser–Ney n-gram model, RNN, LSTM, and causal Transformer on the same Reuters corpus. The task is to predict the next token.

### Files

| File | Purpose |
| --- | --- |
| [part1/data.py](part1/data.py) | Shared loading, splitting, tokenization, vocabulary construction, and encoding. |
| [part1/n_gram.py](part1/n_gram.py) | Train orders 2, 3, and 4; select an order using validation cross-entropy; evaluate and generate text. |
| [part1/rnn.py](part1/rnn.py) | Train, validate, evaluate, and generate text with a unidirectional RNN. |
| [part1/lstm.py](part1/lstm.py) | The corresponding LSTM experiment. |
| [part1/transformer.py](part1/transformer.py) | The corresponding Transformer experiment with causal self-attention. |

### Setup

Create a virtual environment from the repository root:

```sh
python -m venv .venv
```

Activate it using the command for your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux / macOS
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

The neural scripts automatically select CUDA when `torch.cuda.is_available()` is true; otherwise they use CPU. GPU execution requires a compatible CUDA-enabled PyTorch installation. The n-gram implementation runs on CPU.

Check the device before an experiment:

```sh
python -c "import torch; print('PyTorch:', torch.__version__); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Run the experiments

Run these commands individually, from inside `part1`:

```sh
cd part1
python n_gram.py
python rnn.py
python lstm.py
python transformer.py
```

Each script prepares its own copy of the same data, trains its model, selects it using validation loss, evaluates on test data, and prints continuations. Hyperparameters are set in the scripts.

Each model calls `prepare_reuters()` automatically. Running `python data.py` is an optional preparation check that prints split and vocabulary sizes; it does not save encoded data for later processes.

Data preparation calls the NLTK downloader for `reuters`, `punkt`, and `punkt_tab`. Internet access is needed to obtain missing resources. NLTK caches these resources on disk, while the prepared splits and token IDs are rebuilt in memory for each run.

The neural scripts display the loss plot before test evaluation. If a plot window opens, close it to continue. 

### Shared data and evaluation

- Use all 10,788 NLTK Reuters articles, including headlines. Lowercase the raw text and tokenize with `word_tokenize`. THis preserve article boundaries.
- Shuffle article IDs with seed `42`, then split approximately 70/15/15: 7,551 training, 1,618 validation, and 1,619 test articles. 
- Build an uncapped vocabulary from training articles only. Reserve `<unk>`, `<pad>`, `<bos>`, and `<eos>`. Map words absent from the training vocabulary to `<unk>`.
- Encode each article with one leading `<bos>` and one trailing `<eos>`.
- Use neural input windows of 64 tokens, stride 64, and batch size 32. Keep short articles and final partial windows using right padding. Shuffle training batches only.
- Shift targets one position ahead of inputs. Score every target after `<bos>` once, including `<eos>` and `<unk>`, and exclude `<pad>` from loss.
- Report average next-token cross-entropy in natural-log units and perplexity as `exp(cross_entropy)`. Aggregate loss by non-padding target count, not by batch count.

Neural history resets at each window. N-gram evaluation uses the same window boundaries but at most `n - 1` preceding tokens. At article starts, it supplies `n - 1` leading `<bos>` markers to match n-gram training; later window starts are not treated as new articles. All models predict using only preceding tokens.

### Model settings

| Setting | RNN | LSTM | Transformer |
| --- | --- | --- | --- |
| Trainable token embedding width | 128 | 128 | 128 |
| Layers | 3 | 3 | 3 |
| Recurrent hidden width | 256 | 256 | Not applicable |
| Feed-forward internal width | Not applicable | Not applicable | 256 |
| Attention heads | Not applicable | Not applicable | 4 |
| Positions | Recurrent sequence order | Recurrent sequence order | Learned embeddings, maximum length 64 |
| Dropout | 0.2 between recurrent layers | 0.2 between recurrent layers | 0.2 in Transformer blocks |

All neural models use randomly initialized, trainable token embeddings, Adam with learning rate `0.001`, gradient clipping at norm `1.0`, and 30 epochs without early stopping. PyTorch's seed is set to `42`. The checkpoint with the lowest validation cross-entropy is reloaded for testing and generation. The Transformer uses `TransformerEncoderLayer` blocks with an explicit causal mask and a padding mask.

The n-gram script fits interpolated Kneser–Ney models with orders `{2, 3, 4}` and fixed discount `0.75`. `FastKN` precomputes continuation-count tables to speed up scoring. 

The order with the lowest validation cross-entropy is selected for test evaluation and generation. With the uncapped training vocabulary, `<unk>` has no training examples; its n-gram probability comes from the uniform mixture. Document this choice when interpreting results.

### Matched text generation

All four models use the prompts `the company`, `oil prices`, and `the bank`, with 50 new tokens, temperature `1.0`, top-k `20`, and sampling seed `42`. Neural generation uses up to 64 context tokens; n-gram history remains limited to `n - 1`. Generation blocks `<pad>`, `<bos>`, and `<eos>` so that each continuation reaches the requested length; `<unk>` remains allowed.

The n-gram uses NumPy sampling and neural models use PyTorch sampling. 

### Outputs and timing

Files are written to the current working directory (`part1` with the commands above). Re-running an experiment overwrites its checkpoint and plot.

| Model | Best-validation checkpoint | Loss plot |
| --- | --- | --- |
| RNN | `best_rnn_model.pt` | `rnn_loss_curve.png` |
| LSTM | `best_lstm_model.pt` | `lstm_loss_curve.png` |
| Transformer | `best_transformer_model.pth` | `transformer_training_validation_loss.png` |
| N-gram | Kept in memory; not saved | No epoch-based loss plot |

Metrics and generated text are printed to the console, not automatically saved as a results table. 

- **Training time:** neural scripts sum training-loop time across all 30 epochs, excluding validation, preprocessing, checkpoint saving, and plotting. N-gram times are reported separately for each order and include building FastKN tables, excluding subsequent evaluation.
- **Inference time:** one complete test-scoring pass, including iteration/data loading, probability or logit calculation, loss calculation, and progress display. Neural timing includes device transfers and synchronizes CUDA at the timer boundaries. N-gram timing includes building its evaluation cache of repeated context–target losses. These are test-scoring times, not text-generation times; no separate warm-up or repeated timing trials are implemented.
- **Model size:** neural scripts print parameter counts; n-gram prints nonzero stored count entries by order and in total. These are different units, not directly comparable memory measurements. N-gram entry counts exclude the additional FastKN tables and evaluation caches.
