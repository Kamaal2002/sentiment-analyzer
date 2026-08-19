# Sentiment Analyzer

A full-stack sentiment analysis app: a DistilBERT model fine-tuned for binary
sentiment classification on movie reviews, served via a Flask REST API, with
a React frontend. Deployed on Hugging Face Spaces (backend/model) and Vercel
(frontend).

## Project status

Built incrementally, day by day:

- [x] **Day 1** — Dataset & EDA
- [ ] Day 2 — Text preprocessing
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
