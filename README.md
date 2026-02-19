# Sentiment Classification on IMDB: Classical vs Neural Models
## Overview

This project compares classical machine learning approaches and lightweight neural networks for binary sentiment classification on a balanced subset of the IMDB movie review dataset.

The objective is to analyze performance differences between feature-based linear models (TF-IDF + SVM/LogReg/NB) and representation-learning neural baselines (embedding averaging and attention pooling), and to critically evaluate their trade-offs.

## Dataset

We use a balanced subset of the IMDB dataset containing:

- 5,000 movie reviews

- 2,500 positive reviews

- 2,500 negative reviews

The data is split into:

- 70% Training

- 10% Validation

- 20% Test

Validation is used exclusively for hyperparameter tuning. The test set is used only once for final evaluation to avoid data leakage.

## How to Obtain the Dataset

Download the IMDB dataset from:

https://ai.stanford.edu/~amaas/data/sentiment/

After downloading, place the extracted data inside a data/ directory at the root of the repository.

The data/ folder is excluded from version control.

## Methods
### 1. Classical Baselines

We evaluate the following feature representations:

- Bag-of-Words (unigrams)

- TF-IDF (unigrams + bigrams)

Models tested:

- Naive Bayes

- Logistic Regression

- Linear SVM

Hyperparameters are selected using validation accuracy.

The best classical configuration:

**Linear SVM + TF-IDF (1,2), k=20000 features**

### 2. Neural Baselines

Two lightweight PyTorch models were implemented:

**Average Embedding Classifier**

- Learns token embeddings

- Averages non-padding embeddings

- Applies linear classification

**Self-Attention Pooling Classifier**

- Learns a global query vector

- Computes attention scores over tokens

- Produces a weighted document representation

- Provides interpretable attention weights

Neural models are trained with validation-based early stopping.

## Results
| Model	| Test Accuracy |
| :--- | :---|
| SVM + TF-IDF (1,2) | 0.919 |
| AvgEmb (Neural) | 0.881 |
| Attention Pooling (Neural) | 0.867 |

The classical SVM model outperforms both neural baselines.

Confusion matrix analysis shows balanced performance across classes, with similar false positive and false negative rates.

## Analysis

TF-IDF with bigrams performs strongly because sentiment classification relies heavily on lexical signals and short phrases (e.g., “not good”, “highly recommend”). Linear SVMs are well suited to high-dimensional sparse representations.

The neural models learn embeddings from scratch on a relatively small dataset (5k reviews). Without pretrained embeddings and large-scale training, they struggle to surpass strong feature-engineered baselines.

Although the attention pooling model provides token-level interpretability, it does not improve performance over simple embedding averaging in this experiment. This suggests that, given the small dataset and limited training, learning token importance alone is not sufficient to outperform strong lexical baselines.

The lack of improvement from attention pooling highlights that architectural complexity alone does not guarantee better performance. Without sufficient data or sequence modeling capacity, additional parameters may not translate into meaningful gains.

This experiment illustrates the bias–variance trade-off:

- Linear models impose stronger inductive bias and generalize well with limited data.

- Neural models offer higher representational capacity but require more data and tuning.

## Limitations

- Small dataset for training neural embeddings

- No pretrained embeddings

- Limited number of training epochs

- Attention pooling does not capture token interactions

- Simple tokenization (no subword modeling)

## Future Work

- Train on the full IMDB dataset

- Incorporate pretrained embeddings (e.g., GloVe)

- Experiment with sequence models (CNN, LSTM, Transformer encoder)

- Perform broader hyperparameter search

- Analyze calibration and robustness

## How to Run
This project was developed and tested using Visual Studio Code with a Python virtual environment.

**1. Clone the Repository**
```bash
git clone https://github.com/bu-cds-llms/portfolio-piece-1-tgarvia.git
cd portfolio-piece-1-tgarvia
```

**2. Create and Activate Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate # macOS / Linux / WSL
```

On Windows:
```powershell
.\.venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

If using an NVIDIA RTX 50xx GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**4. Open in VS Code**
```bash
code .
```

Make sure to:

- Install the Python extension (Microsoft)

- Install the Jupyter extension (Microsoft)

- Select the `.venv` interpreter:

    - Press `Ctrl+Shift+P`

    - Choose **Python: Select Interpreter**

    - Select `.venv`

**5. Run the Notebook**

Open:
```
notebooks/main_analysis.ipynb
```

Select the `.venv` kernel and click **Run All**.

## Requirements

Key dependencies:

- torch

- scikit-learn

- pandas

- matplotlib

- tqdm

See `requirements.txt` for exact versions.

Repository Structure
```
your-portfolio-piece/
├── README.md
├── requirements.txt
├── notebooks/
│   └── main_analysis.ipynb
├── src/
├── outputs/
└── data/  (not tracked)
```

## Teresa Garvia Gallego

Portfolio Piece – Sentiment Classification
Boston University

##