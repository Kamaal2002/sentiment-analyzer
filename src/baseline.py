"""
TF-IDF + Logistic Regression baseline.

Purpose: give us a real number to compare DistilBERT against. Without a
baseline, "DistilBERT gets 91% accuracy" is not very informative on its own
— we don't know how much of that is the transformer earning its complexity
versus how much a simple linear model over word frequencies would already
get. A classic bag-of-words baseline trains in about a minute on CPU and
tells us exactly that.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_baseline(train_texts, train_labels, max_features: int = 20000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams + bigrams, e.g. "not good" as one feature
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(train_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)

    return vectorizer, clf


def save_baseline(vectorizer, clf, out_dir="../models"):
    joblib.dump(vectorizer, f"{out_dir}/baseline_vectorizer.joblib")
    joblib.dump(clf, f"{out_dir}/baseline_logreg.joblib")


def load_baseline(out_dir="../models"):
    vectorizer = joblib.load(f"{out_dir}/baseline_vectorizer.joblib")
    clf = joblib.load(f"{out_dir}/baseline_logreg.joblib")
    return vectorizer, clf
