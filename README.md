# Sentiment Analyzer

A full-stack sentiment analysis app: a DistilBERT model fine-tuned for binary
sentiment classification on movie reviews, served via a Flask REST API, with
a React frontend. Deployed on Hugging Face Spaces (backend/model) and Vercel
(frontend).

## Project status

Built incrementally, day by day:

- [x] **Day 1** — Dataset & EDA
- [x] **Day 2** — Text preprocessing
- [x] **Day 3** — Baseline model + DistilBERT fine-tuning
- [ ] Day 4 — Interpretability (Captum / attention attribution)
- [ ] Day 5 — REST API
- [ ] Day 6 — Frontend
- [ ] Day 7 — Deployment, polish, docs

## Tech stack

- **Model**: DistilBERT (HuggingFace `transformers`), fine-tuned for binary
  sentiment classification
- **Backend**: Python + Flask REST API
- **Frontend**: React
- **Deployment**: Hugging Face Spaces (backend/model) + Vercel (frontend)

## Project structure

```
sentiment-analyzer/
├── data/           # train/test CSVs, EDA plots (gitignored, regenerate via notebooks/)
├── notebooks/       # exploratory / step-by-step scripts
├── models/         # trained model artifacts (gitignored)
├── src/            # shared preprocessing / training code
├── api/            # Flask REST API
├── frontend/       # React app
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Create and activate a virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Day 1 — Dataset & EDA

Dataset: [IMDB movie reviews](https://huggingface.co/datasets/imdb) via
HuggingFace `datasets` — 25,000 train / 25,000 test reviews, perfectly
balanced between positive and negative labels.

Run the EDA script (cell-marker `.py`, works in VS Code / Jupyter as
notebook cells, or as a plain script):

```bash
cd notebooks
python 01_eda.py
```

This downloads IMDB (cached after first run), prints dataset shape and class
balance, plots review length distributions, and saves clean splits to
`data/train.csv` and `data/test.csv`.

## Day 2 — Text preprocessing

Preprocessing logic lives in [`src/preprocess.py`](src/preprocess.py) so the
training script (Day 3) and the API (Day 4) share identical cleaning +
tokenization.

Key decision: DistilBERT's own tokenizer (WordPiece) already lowercases,
splits punctuation, and breaks words into subwords. Classic NLP steps —
stemming, stopword removal, manual regex tokenizing — are deliberately
**not** used, since they discard information transformer models rely on as
signal. The only manual cleaning is stripping `<br />` HTML artifacts left
over from how IMDB reviews were scraped.

Tokenization uses `max_length=256`. Recomputing truncation with the actual
WordPiece tokenizer (not the word-count estimate from Day 1) showed 40.5% of
reviews get truncated at this length — higher than expected, since subword
tokenization splits many words into multiple tokens. Kept at 256 anyway as a
speed/memory tradeoff for fine-tuning on a free Colab GPU; sentiment is
usually decided early in a review, so losing the tail rarely flips the label.

Run it:

```bash
cd notebooks
python 02_preprocessing.py
```

Saves cleaned text to `data/train_clean.csv` and `data/test_clean.csv`.

## Day 3 — Baseline model + DistilBERT fine-tuning

**Baseline** ([`src/baseline.py`](src/baseline.py),
[`notebooks/03a_baseline.py`](notebooks/03a_baseline.py)): TF-IDF (unigrams +
bigrams) + Logistic Regression. The point of a baseline isn't to be
competitive — it's to establish how much of a transformer's performance is
actually earned versus what a simple linear model over word frequencies
already captures. Bigrams were included specifically so short negations like
"not good" get captured as a feature, avoiding an artificially weak
strawman.

```bash
cd notebooks
python 03a_baseline.py
```

**Fine-tuning** ([`notebooks/03b_finetune_distilbert.py`](notebooks/03b_finetune_distilbert.py)):
`distilbert-base-uncased` fine-tuned for 2 epochs on the IMDB train split.
Meant to run on **Google Colab** with a free GPU — CPU fine-tuning of a
transformer over 25k examples is impractically slow. Trained weights are
saved to `models/distilbert-sentiment/` (gitignored; regenerate via Colab).

**Comparison** ([`notebooks/03c_compare_models.py`](notebooks/03c_compare_models.py))
runs both models on an identical held-out sample and reports results
side by side:

| Model | F1 |
|---|---|
| Baseline (TF-IDF + Logistic Regression) | 0.874 |
| DistilBERT (fine-tuned) | 0.917 |

DistilBERT improves F1 by +4.3 points (+5.0% relative) over the baseline.
The gap is most visible on examples requiring contextual/tonal
understanding — negation, mixed sentiment, sarcasm — rather than simple
keyword matching, which the linear baseline handles reasonably well on its
own already. The set of examples both models get wrong (~4% of test data)
are genuinely ambiguous cases (backhanded compliments, niche context) and
are the starting point for Day 4's interpretability analysis.

Note: CPU inference for DistilBERT over the full 25k-review test set proved
too slow to be practical locally (no GPU on this machine); comparison runs
on a random 3,000-review sample, which is large enough for stable metrics
and closely matches the full-test-set numbers Colab reported during
training.
