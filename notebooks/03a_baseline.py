# %% [markdown]
# # Baseline: TF-IDF + Logistic Regression
#
# Before trusting DistilBERT's accuracy number, it's worth having a
# reference point — how well does a simple, fast linear model do on the
# same task? Sentiment has a lot of surface-level lexical signal ("terrible",
# "brilliant"), so a bag-of-words model should already do reasonably well.
# This tells us how much DistilBERT is actually earning over that.

# %%
import sys

sys.path.append("../src")

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save plots to file, don't try to display them

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from baseline import save_baseline, train_baseline

# %%
train_df = pd.read_csv("../data/train_clean.csv")
test_df = pd.read_csv("../data/test_clean.csv")
print("Train:", train_df.shape, " Test:", test_df.shape)

# %% [markdown]
# TF-IDF converts each review into a sparse vector of word/bigram
# importance scores; logistic regression learns a linear boundary over
# those features. Trains in under a minute on CPU.

# %%
vectorizer, clf = train_baseline(train_df["text"], train_df["label"])
print("Baseline trained. Vocabulary size:", len(vectorizer.vocabulary_))

# %%
X_test = vectorizer.transform(test_df["text"])
preds = clf.predict(X_test)

print(classification_report(test_df["label"], preds, target_names=["Negative", "Positive"]))

# %%
cm = confusion_matrix(test_df["label"], preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Baseline (TF-IDF + LogReg) — Confusion Matrix")
plt.tight_layout()
plt.savefig("../data/baseline_confusion_matrix.png", dpi=120)

# %% [markdown]
# ## Misclassified examples
#
# Quick look at what trips it up — negation, sarcasm, mixed sentiment are
# the usual suspects for a bag-of-words model.

# %%
test_df["pred"] = preds
wrong = test_df[test_df["label"] != test_df["pred"]]
print(f"{len(wrong)} misclassified out of {len(test_df)} ({len(wrong)/len(test_df)*100:.1f}%)\n")

for _, row in wrong.sample(3, random_state=1).iterrows():
    true_label = "Positive" if row["label"] == 1 else "Negative"
    pred_label = "Positive" if row["pred"] == 1 else "Negative"
    print(f"True: {true_label} | Predicted: {pred_label}")
    print(row["text"][:250], "...\n")

# %%
save_baseline(vectorizer, clf)
print("Saved models/baseline_vectorizer.joblib and models/baseline_logreg.joblib")
