# Sentiment Analyzer

A full-stack sentiment analysis app: a DistilBERT model fine-tuned for binary
sentiment classification on movie reviews, served via a Flask REST API, with
a React frontend. Deployed on Hugging Face Spaces (backend/model) and Vercel
(frontend).

## Project status

Built incrementally, day by day:

- [x] **Day 1** — Dataset & EDA
- [x] **Day 2** — Text preprocessing
- [ ] Day 3 — Model fine-tuning
- [ ] Day 4 — REST API
- [ ] Day 5 — Frontend
- [ ] Day 6 — Deployment
- [ ] Day 7 — Polish, testing, docs

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
