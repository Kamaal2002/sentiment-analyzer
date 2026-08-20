import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_baseline(train_texts, train_labels, max_features: int = 20000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams + bigrams, so "not good" survives as one feature
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
