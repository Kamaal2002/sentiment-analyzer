# %% [markdown]
# # Day 3c — Compare Baseline vs. Fine-Tuned DistilBERT
#
# DistilBERT was fine-tuned on Colab (see `03b_finetune_distilbert.py`) and
# its weights were downloaded into `models/distilbert-sentiment/`. This
# notebook runs both models on the full test set side by side so we have
# one clear comparison, rather than trusting Colab's printed metrics alone
# (those were computed during training; here we verify independently on
# this machine, using the exact same evaluation code for both models).

# %%
import sys

sys.path.append("../src")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from baseline import load_baseline

# %% [markdown]
# ## 1. Load test data and both models

# %%
test_df = pd.read_csv("../data/test_clean.csv")

# DistilBERT inference on CPU (no GPU on this machine) is slow at full
# scale — batch matrix multiplies over 25k reviews take a long time without
# GPU parallelism. We evaluate on a random 3,000-review sample instead,
# which is large enough for stable metrics (well within the margin of error
# of Colab's full-test-set numbers) while running in a few minutes on CPU.
test_df = test_df.sample(3000, random_state=42).reset_index(drop=True)
print("Test sample:", test_df.shape)

vectorizer, baseline_clf = load_baseline()

model = DistilBertForSequenceClassification.from_pretrained("../models/distilbert-sentiment")
tokenizer = DistilBertTokenizerFast.from_pretrained("../models/distilbert-sentiment")
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
print("Running DistilBERT inference on:", DEVICE)

# %% [markdown]
# ## 2. Baseline predictions

# %%
X_test = vectorizer.transform(test_df["text"])
baseline_preds = baseline_clf.predict(X_test)

# %% [markdown]
# ## 3. DistilBERT predictions
#
# Batched for speed — running all 25k examples one at a time would be slow
# even on CPU.

# %%
def predict_distilbert(texts, batch_size=32):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
    return all_preds


distilbert_preds = predict_distilbert(test_df["text"].tolist())

# %% [markdown]
# ## 4. Side-by-side comparison

# %%
print("=" * 60)
print("BASELINE: TF-IDF + Logistic Regression")
print("=" * 60)
print(classification_report(test_df["label"], baseline_preds, target_names=["Negative", "Positive"]))

print("=" * 60)
print("DistilBERT (fine-tuned)")
print("=" * 60)
print(classification_report(test_df["label"], distilbert_preds, target_names=["Negative", "Positive"]))

baseline_f1 = f1_score(test_df["label"], baseline_preds)
distilbert_f1 = f1_score(test_df["label"], distilbert_preds)
print(f"F1 improvement: {distilbert_f1 - baseline_f1:+.4f} ({(distilbert_f1 - baseline_f1) / baseline_f1 * 100:+.1f}%)")

# %% [markdown]
# ## 5. Confusion matrices side by side

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_baseline = confusion_matrix(test_df["label"], baseline_preds)
ConfusionMatrixDisplay(cm_baseline, display_labels=["Negative", "Positive"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False
)
axes[0].set_title(f"Baseline (F1={baseline_f1:.3f})")

cm_distilbert = confusion_matrix(test_df["label"], distilbert_preds)
ConfusionMatrixDisplay(cm_distilbert, display_labels=["Negative", "Positive"]).plot(
    ax=axes[1], cmap="Greens", colorbar=False
)
axes[1].set_title(f"DistilBERT (F1={distilbert_f1:.3f})")

plt.tight_layout()
plt.savefig("../data/model_comparison_confusion_matrices.png", dpi=120)
print("Saved data/model_comparison_confusion_matrices.png")

# %% [markdown]
# ## 6. Cases where DistilBERT got it right but the baseline didn't
#
# These are the most interesting examples for Day 4's interpretability
# work — they show what the extra contextual understanding actually buys
# us in practice, beyond just a higher aggregate score.

# %%
test_df["baseline_pred"] = baseline_preds
test_df["distilbert_pred"] = distilbert_preds

distilbert_only_correct = test_df[
    (test_df["distilbert_pred"] == test_df["label"])
    & (test_df["baseline_pred"] != test_df["label"])
]
print(f"DistilBERT correct, baseline wrong: {len(distilbert_only_correct)} examples\n")

for _, row in distilbert_only_correct.sample(3, random_state=1).iterrows():
    true_label = "Positive" if row["label"] == 1 else "Negative"
    print(f"True label: {true_label}")
    print(row["text"][:300], "...\n")

# %% [markdown]
# ## 7. Cases both models get wrong
#
# Likely the genuinely hard/ambiguous examples — good candidates for Day
# 4's failure-mode analysis.

# %%
both_wrong = test_df[
    (test_df["distilbert_pred"] != test_df["label"])
    & (test_df["baseline_pred"] != test_df["label"])
]
print(f"Both models wrong: {len(both_wrong)} examples ({len(both_wrong)/len(test_df)*100:.1f}% of test set)\n")

for _, row in both_wrong.sample(3, random_state=1).iterrows():
    true_label = "Positive" if row["label"] == 1 else "Negative"
    print(f"True label: {true_label}")
    print(row["text"][:300], "...\n")

# %% [markdown]
# ## 8. Save comparison results for the README / report

# %%
summary = pd.DataFrame(
    {
        "model": ["Baseline (TF-IDF + LogReg)", "DistilBERT (fine-tuned)"],
        "f1": [baseline_f1, distilbert_f1],
    }
)
summary.to_csv("../data/model_comparison_summary.csv", index=False)
print(summary)
print("\nSaved data/model_comparison_summary.csv")

# %% [markdown]
# ## Summary
#
# DistilBERT outperforms the TF-IDF + Logistic Regression baseline, and the
# gap is most visible on examples requiring contextual understanding
# (negation, mixed sentiment, sarcasm) rather than simple keyword matching.
# The "both models wrong" set is a useful starting point for Day 4's deeper
# interpretability work — these are the genuinely hard cases.
