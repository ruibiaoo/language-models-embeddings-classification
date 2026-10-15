# DSA4213 Assignment 1

## 🧐 About the Project
This repository contains the implementation for DSA4213 Assignment 1, which investigates how different language modelling architectures and word-embedding strategies affect language-model performance and learned text representations.
The project currently covers:
- Part I: Compare an interpolated Kneser-Ney n-gram model, RNN, LSTM, and causal Transformer for language modelling.
- Part II: Investigate the effects of randomly initialized trainable embeddings, self-trained frozen Skip-gram embeddings, and publicly pretrained Word2Vec embeddings on LSTM language model performance.
- Part III:  Evaluate and compare the effectiveness of learned language model representations for downstream text classification

Main technologies
- PyTorch — Neural language models and training
- Hugging Face Datasets — TinyStories and AG News Dataset
- NLTK — Tokenization and n-gram language modelling
- Gensim — Word2Vec / Skip-gram embeddings
- NumPy — Numerical operations 
- Matplotlib — Visual Plots
- tqdm — Progress bars
- scikit-learn — Logistic regression and classification evaluation

## 🚀 Getting Started

### Prerequisites

Ensure the following are installed before running the project:

| Requirement | Version / Notes |
| --- | --- |
| Python | 3.12 recommended |
| pip | Latest version recommended |
| Internet connection | Required to download TinyStories, AG News, NLTK resources, and the Google News Word2Vec embeddings |
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

Parts I & II: TinyStories
Parts I and II use [roneneldan/TinyStories on Hugging Face ](https://huggingface.co/datasets/roneneldan/TinyStories). TinyStories contains synthetic English stories generated using GPT-3.5 and GPT-4; see the [dataset paper](https://arxiv.org/abs/2305.07759) here.

The data loaders use:

```python
datasets.load_dataset(..., streaming=True)
```

Therefore, no manual dataset download or corpus placement is required. However, internet access is needed to stream data and obtain missing NLTK resources. 

Part III uses [fancyzhx/ag_news on Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news). AG News is a widely used benchmark dataset in Natural Language Processing (NLP) for text classification and topic modeling, and consists of 4 classes - World, Sports, Business and Sci/Tech; see the [dataset paper](https://arxiv.org/pdf/1509.01626) here.

The data loaders use:

```python
load_dataset(...)
```
Similarly, internet access is needed to obtain missing NLTK resources.

### Data split

#### Parts I & II: TinyStories

The TinyStories dataset is used to train and evaluate different language modelling architectures and embedding strategies. The preprocessing pipeline selects **24,000 stories** across the following splits:

| Split | Number of Stories |
|---|---:|
| Training | 20,000 |
| Validation | 2,000 |
| Test | 2,000 |

#### Part III: AG News

The AG News dataset is used for downstream text classification. The preprocessing pipeline selects **67,600 articles** across four equally represented categories.

Training and validation subsets are sampled from the original training split, while the original test split is retained for final evaluation.

| Category | Training | Validation | Test |
|---|---:|---:|---:|
| World | 12,500 | 2,500 | 1,900 |
| Sports | 12,500 | 2,500 | 1,900 |
| Business | 12,500 | 2,500 | 1,900 |
| Sci/Tech | 12,500 | 2,500 | 1,900 |
| **Total** | **50,000** | **10,000** | **7,600** |


For TinyStories, the source training and validation splits are independently shuffled using:

- **Random seed:** `42`
- **Shuffle buffer size:** `10,000`


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

[part3/src/data.py](part3/src/data.py) prepares AG News for both classification experiments:

1. Load `fancyzhx/ag_news`, including its original training and official test splits.
2. Lowercase text and normalize whitespace. Exclude empty or duplicate articles from the training pool, and exclude training articles whose normalized text matches an official test article.
3. Using seed **42**, select **50,000 training** and **10,000 validation** articles from the cleaned training pool, with equal numbers from each of the four classes. Retain all **7,600 official test** articles.
4. Tokenize articles using NLTK `word_tokenize()`, retaining punctuation and numbers.
5. Encode articles using the vocabulary saved with the corresponding Part II LSTM. Do not build a new vocabulary from AG News; words absent from the Part II vocabulary map to `<unk>`.
6. For each classifier input, keep at most **62 article tokens**, add `<bos>` and `<eos>`, and pad the sequence to **64 positions**.

Each model script calls `prepare_dataset()` itself, so running a `data.py` file beforehand is optional. To inspect the data splits and statistics without training a model:

```sh
python part1/src/data.py
python part2/src/data.py
python part3/src/data.py
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
python part2/src/word2vec-pretrained-embedding.py
```

Part II compares the following embedding strategies using the same LSTM language model architecture:

- Random trainable embeddings
- Self-trained frozen Skip-gram embeddings
- Public pretrained frozen Word2Vec embeddings


### Part III: Downstream Representation Analysis

The classification scripts require these files in `part2/best_models/`:

- `best_lstm_model_random_embedding.pt` and `lstm_random_vocab.json`
- `best_lstm_model_pretrained_word2vec_embedding.pt` and `lstm_pretrained_vocab.json`

Run the corresponding Part II experiments first. If the vocabulary JSON files are missing, generate them with:

```sh
python part2/src/export_vocab.py
```

The exported vocabulary must match the vocabulary used to train the saved checkpoints.

Run the downstream classification experiments using:

```sh
python part3/src/lstm_random_embedding_classification.py
python part3/src/lstm_pretrained_embedding_classification.py
```

Part III compares the downstream text classification performance of two LSTM language models pretrained in Part II:

- Random Embeddings: LSTM trained with randomly initialized, trainable embeddings.
- Pretrained Word2Vec Embeddings: LSTM trained with frozen, publicly pretrained Word2Vec embeddings.


### Experiment Workflow

Each Part I and Part II script runs a complete language-model experiment:

1. Prepare the dataset and vocabulary.
2. Train the model.
3. Select the best checkpoint based on validation loss.
4. Evaluate the selected model on the test set.
5. Generate text using the predefined prompts and sampling configuration.

Each Part III script runs a complete downstream classification experiment:

1. Load the saved Part II LSTM checkpoint and its matching vocabulary.
2. Prepare the AG News splits and encode articles using that vocabulary.
3. Freeze the LSTM and extract one pooled representation per article.
4. Train logistic-regression classifiers with different `C` values and select the best using validation macro-F1.
5. Evaluate the selected classifier on the official test set.
6. Save the test metrics, article-level predictions, and confusion matrix 


Model settings and hyperparameters are defined directly in the scripts rather than supplied through command-line arguments. 

When comparing training or inference times, run experiments sequentially to reduce interference from concurrent workloads.


### Saving Experiment Logs

Part I and Part II console output contains key experimental information including:

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

Part III prints classifier validation and test metrics. Each classification script also saves test metrics, article-level predictions, and a confusion matrix in part3/results/.

### Headless Execution

Training-loss plots are displayed during execution. Close the plot window to continue to test evaluation.


## ⚙️ Model Configurations
### Shared Language Model Training and Evaluation
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

Loss is averaged over all non-padding targets. Every target after `<bos>` is scored once, including `<eos>` and `<unk>`; `<pad>` is ignored. Test evaluation and generation reload the best-validation neural checkpoint rather than using the final epoch automatically.

All predictions use preceding context only. RNN/LSTM hidden states reset at each window. The Transformer uses a causal attention mask and padding mask. N-gram evaluation uses the same target tokens and window boundaries, with at most `n - 1` preceding tokens. Story starts are padded to `n - 1` `<bos>` markers for n-grams; later window boundaries do not receive extra story-start markers.

### Part I Neural Configurations
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

### Part II Embedding Configurations
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
| Public Word2Vec | Google News `word2vec-google-news-300` vectors | No; the entire input table is frozen |

For the self-trained Skip-gram variant, embeddings are trained using Gensim Word2Vec on the training split only.

For the public Google News Word2Vec variant, ordinary vocabulary words without a pretrained vector retain their randomly initialized vectors. `<bos>`, `<eos>`, and `<unk>` also retain random vectors, while `<pad>` is zero. The entire input-embedding table remains frozen.

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

For the self-trained Skip-gram variant, embeddings are trained using only the training split and are frozen during subsequent LSTM training. Tokens not covered by the learned embeddings, including `<bos>`, `<eos>`, and `<unk>`, retain their randomly initialized vectors, while `<pad>` is set to zero.

Embedding coverage is reported as the proportion of training-vocabulary words with available Skip-gram vectors. Word2Vec pretraining time is measured separately from LSTM training time to distinguish embedding-learning cost from language-model training cost.


### Part III: Downstream Classification Configurations

Part III compares the downstream classification performance of LSTM language models from Part II using random trainable embeddings and pretrained Word2Vec embeddings on the AG News dataset.

The pretrained LSTMs are frozen and used as feature extractors. Only the logistic regression classifiers are trained.

#### Feature Extraction

| Setting | Value |
|---|---|
| Feature extractor | Pretrained LSTM from Part II |
| Embedding dimension | 300 |
| LSTM hidden dimension | 256 |
| LSTM layers | 3 |
| Maximum sequence length | 64 |
| Batch size | 32 |
| Feature extraction | Mean pooling of LSTM hidden states |
| Pooling exclusions | PAD, BOS, EOS |
| Output feature dimension | 256 |

#### Classifier Training and Evaluation

| Setting | Value |
|---|---|
| Classifier | Logistic Regression |
| Feature scaling | StandardScaler |
| Solver | `lbfgs` |
| Regularization C | 0.001, 0.01, 0.1, 1, 10, 100 |
| Maximum iterations | 1,000 |
| Random seed | 42 |
| Model selection | Highest validation macro-F1 |
| Test metrics | Accuracy, macro-F1, per-class classification report |

The best classifier is selected using validation macro-F1 and evaluated on the held-out test set. Both embedding variants use the same classification procedure.

## 🔁 Reproducibility
The main random seed used throughout the project is: 42

| Component | Setting |
| --- | --- |
| TinyStories buffered shuffle | Seed 42 |
| PyTorch model initialization | `torch.manual_seed(42)`|
| Neural generation | Device-local generator seeded with 42 |
| N-gram generation | `np.random.default_rng(42)`|
| Skip-gram training | Gensim seed 42|
| AG News stratified training/validation sampling | NumPy seed 42 |

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
│   │   ├── word2vec-pretrained-embedding.py
│   │   └── export_vocab.py
│   ├── best_models/
│   │   ├── lstm_random_vocab.json
│   │   ├── lstm_pretrained_vocab.json
│   │   ├── best_lstm_model_random_embedding.pt
│   │   ├── best_lstm_model_skipgram_embedding.pt
│   │   └── best_lstm_model_pretrained_word2vec_embedding.pt
│   └── results/
│       ├── lstm_random_300_loss_curve.png
│       ├── lstm_skipgram_300_loss_curve.png
│       └── lstm_pretrained_word2vec_300_loss_curve.png
│
├── part3/
│   ├── src/
│   │   ├── data.py
│   │   ├── lstm_pretrained_embedding_classification.py
│   │   └── lstm_random_embedding_classification.py
│   └── results/
│       ├── lstm_pretrained_metrics.json
│       ├── lstm_pretrained_predictions.csv
│       ├── lstm_pretrained_confusion_matrix.png
│       ├── lstm_random_metrics.json
│       ├── lstm_random_predictions.csv
│       └── lstm_random_confusion_matrix.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 📄 License

This repository was developed for DSA4213 Assignment 1 at the National University of Singapore (NUS) and is intended for academic and educational purposes.

The code is not currently released under an open-source license. External datasets, pretrained models, and libraries remain subject to their respective licenses and terms of use.
