# DSA4213 Assignment 1

## 🧐 About the Project
This repository contains the implementation for DSA4213 Assignment 1, which investigates how different language modelling architectures and word-embedding strategies affect language-model performance and learned text representations.
The project currently covers:
- Part I: comparison of an interpolated Kneser-Ney n-gram model, RNN, LSTM, and causal Transformer with self-attention.
- Part II: comparison of LSTM language models using random trainable embeddings, self-trained frozen Skip-gram embeddings, and a public pretrained Word2Vec embedding.
- Part III: 

Main technologies
- PyTorch — Neural language models and training
- Hugging Face Datasets — TinyStories Dataset
- NLTK — Tokenization and n-gram language modelling
- Gensim — Word2Vec / Skip-gram embeddings
- NumPy — Numerical operations 
- Matplotlib — Visual Plots
- tqdm — progress bars

## 🚀 Getting Started

### Prerequisites

Ensure the following are installed before running the project:

| Requirement | Version / Notes |
| --- | --- |
| Python | 3.12 recommended |
| pip | Latest version recommended |
| Internet connection | Required for TinyStories streaming and NLTK resource downloads |
| CUDA-compatible GPU | Optional; used automatically by PyTorch when available |

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ruibiaoo/dsa4213_assignment
   cd dsa4213_assignment
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate the environment**

   **Windows PowerShell**

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **Linux / macOS**

   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

   To record PyTorch and device information:

   ```bash
   python -c "import platform, torch; print(platform.platform()); print('CPU:', platform.processor()); print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

## 📚 Dataset and Preprocessing

Parts I and II use [roneneldan/TinyStories on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories). TinyStories contains synthetic English stories generated using GPT-3.5 and GPT-4; see the [dataset paper](https://arxiv.org/abs/2305.07759) here.

The data loaders use:

```python
datasets.load_dataset(..., streaming=True)
```

Therefore, no manual dataset download or corpus placement is required. However, internet access is needed to stream data and obtain missing NLTK resources. 

### Data split

The current preprocessing pipeline selects:

| Split | Number of stories |
| --- | ---: |
| Training | 20,000 |
| Validation | 2,000 |
| Test | 2,000 |

The source training split and validation split are independently shuffled using:

- Seed: `42`
- Shuffle buffer: `10,000`

Using the same seed and buffer size ensures that the streaming dataset follows the same deterministic shuffling procedure when the dataset revision and software environment are unchanged. Since TinyStories is accessed through Hugging Face streaming, exact reproduction also depends on using the same dataset revision, preprocessing code, tokenizer/package versions, and vocabulary construction procedure.

### Preprocessing pipeline

[part1/src/data.py](part1/src/data.py) and [part2/src/data.py](part2/src/data.py) contain identical preparation code:

1. Stream the source `train` split using buffered shuffling with seed **42** and buffer size **10,000**. This is buffered shuffling, not a uniform shuffle of the entire corpus.
2. Normalize whitespace and lowercase each row's text. Exclude empty stories. Retain punctuation and numbers.
3. Remove exact duplicates of normalized text using SHA-256 fingerprints. A shared set prevents duplicates both within and across selected splits and it does not remove near-duplicates.
4. Tokenize each accepted story using NLTK `word_tokenize()`, retaining story boundaries.
5. Select **20,000 training stories**. Separately shuffle the source `validation` split with the same seed/buffer settings and select **4,000 additional unique stories**: the first 2,000 become validation and the next 2,000 become test.
6. Build the vocabulary exclusively from training tokens, with no frequency cutoff or vocabulary-size cap. Assign `<unk>=0`, `<pad>=1`, `<bos>=2`, and `<eos>=3`; ordinary tokens follow training-frequency order.
7. Encode each story as `[<bos>, token IDs..., <eos>]`. Validation/test words absent from the training vocabulary map to `<unk>`.

With this vocabulary policy, ordinary training words are never replaced by `<unk>`, so it has no training examples. Vocabulary size, text-token counts, and unknown-token rates are printed during preparation. Use values from the actual run rather than hard-coding an earlier vocabulary size.

Each model calls `prepare_dataset()` itself. Running `data.py` first is optional. To inspect data statistics without training:

```sh
python part1/src/data.py
python part2/src/data.py
```

## 💻 Usage

Run scripts by their file paths directly from the repository root. 


### Part I: Language Models Training and Comparison

Run each language model experiment using:

```sh
python part1/src/n_gram.py
python part1/src/rnn.py
python part1/src/lstm.py
python part1/src/transformer.py
```
Part I compares the following language modelling approaches:

- Interpolated Kneser-Ney n-gram
- Recurrent Neural Network (RNN)
- Long Short-Term Memory network (LSTM)
- Causal Transformer with self-attention

### Part II: Embedding Variants and Ablation

Run each embedding experiment using:

```sh
python part2/src/random-embedding.py
python part2/src/skip-gram-embedding.py
```

Part II compares the following embedding strategies using the same LSTM language model architecture:

- Random trainable embeddings
- Self-trained frozen Skip-gram embeddings
- Public pretrained frozen Word2Vec embeddings


### Part III: Downstream Representation Analysis

Run the downstream representation experiments using:




### Experiment Workflow

Each Part I and Part II script runs a complete language-model experiment:

1. Prepare the dataset and vocabulary.
2. Train the model.
3. Select the best checkpoint based on validation loss.
4. Evaluate the selected model on the test set.
5. Generate text using the predefined prompts and sampling configuration.

Model settings and hyperparameters are defined directly in the scripts rather than supplied through command-line arguments. 

When comparing training or inference times, run experiments sequentially to reduce interference from concurrent workloads.


### Saving Experiment Logs

Console Output contains key experimental information including:

- Training and validation losses
- Test metrics
- Training and inference times
- Embedding coverage, where applicable
- Generated text samples

To save the output of an experiment:
```sh
python -u part1/src/lstm.py > part1/lstm_run.log 2>&1
```

or:
```sh
python -u part2/src/skip-gram-embedding.py > part2/skipgram_run.log 2>&1
```

### Headless Execution

Training-loss plots are displayed during execution. Close the plot window to continue to test evaluation.


## ⚙️ Model Configurations
## Shared Language Model Training and Evaluation
Neural models are trained using the shared settings below, while the n-gram model is evaluated using the same tokenized data, target tokens, and test split.
| Setting | Value |
| --- | --- |
| Input window length / stride | 64 / 64 |
| Batch size | 32 |
| Epochs| 50 |
| Optimizer | Adam |
| Learning Rate| 0.001 |
| Gradient clipping | Maximum norm 1.0 |
| Checkpoint selection | Lowest validation cross-entropy |
| Test metrics | Average next-token cross-entropy and perplexity |
| Padding | Ignored in loss calculation |

Loss is averaged over all non-padding targets.. Every target after `<bos>` is scored once, including `<eos>` and `<unk>`; `<pad>` is ignored. Test evaluation and generation reload the best-validation neural checkpoint rather than using the final epoch automatically.

All predictions use preceding context only. RNN/LSTM hidden states reset at each window. The Transformer uses a causal attention mask and padding mask. N-gram evaluation uses the same target tokens and window boundaries, with at most `n - 1` preceding tokens. Story starts are padded to `n - 1` `<bos>` markers for n-grams; later window boundaries do not receive extra story-start markers.

## Part I Neural Configurations
The RNN, LSTM, and Transformer use comparable embedding dimensions, layer counts, and training settings so that differences in performance can be interpreted primarily in terms of model architecture. Their architecture-specific configurations are summarized below.

| Setting | RNN | LSTM | Transformer |
| --- | --- | --- | --- |
| Random trainable input embedding dimension | 128 | 128 | 128 |
| Layers | 3 | 3 | 3 |
| Recurrent hidden dimension | 256 | 256 | N/A |
| Feed-forward internal dimension | N/A | N/A | 256 |
| Attention heads | N/A | N/A | 4 |
| Position representation | Recurrent order | Recurrent order | Learned embeddings, maximum length 64 |
| Dropout | 0.2 between recurrent layers | 0.2 between recurrent layers | 0.2 in Transformer blocks |
| Activation |  `tanh` |  LSTM gates | `relu`  |

The Transformer uses a causal attention mask, so predictions depend only on preceding context.

The n-gram model uses NLTK interpolated Kneser-Ney smoothing. Candidate orders are **2, 3, and 4**, with discount **0.75**. `FastKN` precomputes continuation-count tables to improve scoring efficiency. Evaluation and generation use a uniform mixture:

```text
P(word | history) = 0.999 * P_KN(word | history) + 0.001 / vocabulary_size
```

`uniform_weight=0.001` supplies nonzero probabilities, including for `<unk>`, which has no training counts. 
The order with the lowest validation cross-entropy is selected for testing.

## Part II Embedding Configurations
Part II investigates how different word-embedding strategies affect LSTM language-model performance. To ensure a fair comparison, the LSTM architecture, training procedure, dataset, and evaluation settings are kept constant across all embedding variants. The main experimental variable is therefore the initialization and trainability of the input embedding layer.

The shared LSTM configuration used for all Part II experiments is shown below: 

| Setting | Value |
| --- | --- |
| Embedding dimension | 300 | 
| Hidden dimension | 256 | 
| LSTM layers | 3 | 
| Dropout| 0.2 | 

Three embedding strategies are compared. They differ in how the embedding vectors are obtained and whether they are updated during LSTM training:

| Variant | Initialization | Updated during LSTM training? |
| --- | --- | --- |
| Random | PyTorch `nn.Embedding` random initialization; zero padding row | Yes, except padding |
| Self-trained Skip-gram | Word2Vec vectors fitted to training stories | No; the entire input table is frozen |
| Public Word2Vec ) | Google News `word2vec-google-news-300` vectors | No; the entire input table is frozen |

For the self-trained Skip-gram variant, embeddings are trained using Gensim Word2Vec on the training split only. 

The Skip-gram model is trained using the following configuration:

| Parameter | Value |
| --- | --- |
| `vector_size` | 300 |
| `sg` | 1 (Skip-gram) |
| `window` | 5 |
| `min_count` | 1 |
| `negative` | 5 |
| `epochs` | 20 |
| `workers` | 4 CPU workers |
| `seed` | 42 |

For the self-trained Skip-gram variant, embeddings are trained using only the training split and are frozen during subsequent LSTM training. Tokens not covered by the learned embeddings, including <bos>, <eos>, and <unk>, retain their randomly initialized vectors, while <pad> is set to zero.

Embedding coverage is reported as the proportion of training-vocabulary words with available Skip-gram vectors. Word2Vec pretraining time is measured separately from LSTM training time to distinguish embedding-learning cost from language-model training cost.

## 🔁 Reproducibility
The main random seed used throughout the project is:

```sh
42
``` 

| Component | Setting |
| --- | --- |
| TinyStories buffered shuffle | Seed 42 |
| PyTorch model initialization | `torch.manual_seed(42)`|
| Neural generation | Device-local generator seeded with 42 |
| N-gram generation | `np.random.default_rng(42)`|
| Skip-gram training | Gensim seed 42|

### Reproducibility limitations
Exact reproduction is not guaranteed from the seed alone because:
- TinyStories is streamed rather than archived locally.
- The Hugging Face dataset revision is not currently pinned.
- The selected processed splits and vocabulary are rebuilt for each process.
- Requirements are currently not pinned to exact historical package versions.
- Multi-worker Word2Vec training can introduce scheduling variation.
- GPU results may vary across hardware and software environments.

## 📁 Project Structure

A simplified repository structure is:

```text
├── part1/
│   ├── src/
│   │   ├── data.py
│   │   ├── n_gram.py
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   ├── best_models/
│   │   ├── best_4_gram_model.pkl
│   │   ├── best_lstm_model.pt
│   │   ├── best_rnn_model.pt
│   │   └── best_transformer_model.pth
│   └── results/
│       ├── lstm_loss_curve.png
│       ├── rnn_loss_curve.png
│       └── transformer_loss_curve.png
│
├── part2/
│   ├── src/
│   │   ├── data.py
│   │   ├── random-embedding.py
│   │   ├── skip-gram-embedding.py
│   │   └── word2vec-pretrained-embedding.py
│   ├── best_models/
│   │   ├── best_lstm_model_random_embedding.pt
│   │   ├── best_lstm_model_skipgram_embedding.pt
│   │   └── 
│   └── results/
│       ├── lstm_random_300_loss_curve.png
│       ├── lstm_skipgram_300_loss_curve.png
│       └── 
│    
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🙏 References and Acknowledgments

### Dataset and documentation

- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [TinyStories paper](https://arxiv.org/abs/2305.07759)
- [NLTK language-model documentation](https://www.nltk.org/api/nltk.lm.html)
- [Gensim Word2Vec documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [Gensim pretrained model catalogue](https://github.com/piskvorky/gensim-data#models)

---

## 📄 License

This repository is primarily intended for **DSA4213 coursework**.
