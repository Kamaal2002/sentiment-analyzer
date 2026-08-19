# %% [markdown]
# # Day 1 — Dataset & Exploratory Data Analysis
#
# Goal: load the IMDB movie review dataset, understand its shape and balance,
# look at review length distributions, and save clean train/test CSVs that
# later steps (preprocessing, fine-tuning) will build on.

# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load the dataset
#
# We use the `datasets` library from HuggingFace, which downloads and caches
# IMDB automatically (50,000 movie reviews, labeled positive/negative).
# It's the standard benchmark dataset for binary sentiment classification —
# well known, cleanly balanced, and small enough to iterate on quickly.

# %%
raw = load_dataset("imdb")
print(raw)

# %% [markdown]
# `raw` has `train` (25k), `test` (25k), and `unsupervised` (50k, unlabeled —
# we won't use that split). Each example is a dict with `text` and `label`
# (0 = negative, 1 = positive).

# %%
train_df = raw["train"].to_pandas()
test_df = raw["test"].to_pandas()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# %% [markdown]
# ## 2. Class balance
#
# Before training anything, we check whether positive/negative labels are
# roughly balanced. An imbalanced dataset would need extra handling (class
# weights, resampling) or accuracy alone would be a misleading metric.

# %%
print("Train label counts:")
print(train_df["label"].value_counts())
print("\nTest label counts:")
print(test_df["label"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, df, title in zip(axes, [train_df, test_df], ["Train", "Test"]):
    counts = df["label"].value_counts().sort_index()
    ax.bar(["Negative", "Positive"], counts.values, color=["#e07a5f", "#3d5a80"])
    ax.set_title(f"{title} class balance")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("../data/class_balance.png", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Sample reviews
#
# Reading a few raw examples helps you understand what the model will
# actually see — HTML artifacts like `<br />` line breaks are common in this
# dataset and worth knowing about now (we'll clean them up on Day 2).

# %%
for i in range(3):
    label = "Positive" if train_df.iloc[i]["label"] == 1 else "Negative"
    print(f"--- Example {i} ({label}) ---")
    print(train_df.iloc[i]["text"][:300], "...\n")

# %% [markdown]
# ## 4. Review length / word count distribution
#
# Understanding review length matters directly for Day 3: DistilBERT has a
# max input length (typically 512 tokens), so we need to know how many
# reviews would get truncated, and pick a sensible `max_length` for tokenization.

# %%
train_df["word_count"] = train_df["text"].str.split().str.len()
test_df["word_count"] = test_df["text"].str.split().str.len()

print("Train word count stats:")
print(train_df["word_count"].describe())

# %%
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(train_df["word_count"], bins=50, kde=True, ax=ax, color="#3d5a80")
ax.axvline(train_df["word_count"].median(), color="red", linestyle="--", label="median")
ax.set_title("Distribution of review word counts (train)")
ax.set_xlabel("Word count")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("../data/word_count_distribution.png", dpi=120)
plt.show()

# %%
# How many reviews exceed common truncation thresholds?
for threshold in [128, 256, 512]:
    pct = (train_df["word_count"] > threshold).mean() * 100
    print(f"Reviews over {threshold} words: {pct:.1f}%")

# %% [markdown]
# ## 5. Save cleaned train/test splits
#
# We keep this "clean" step minimal on Day 1 — just the raw text and label,
# with the helper `word_count` column dropped. Real text cleaning
# (lowercasing, HTML stripping, tokenization) is Day 2's job. Splitting and
# saving now gives every later script (preprocessing, training, API) a
# single consistent source of truth to read from.

# %%
os.makedirs("../data", exist_ok=True)

train_out = train_df[["text", "label"]]
test_out = test_df[["text", "label"]]

train_out.to_csv("../data/train.csv", index=False)
test_out.to_csv("../data/test.csv", index=False)

print("Saved:")
print(" data/train.csv ->", train_out.shape)
print(" data/test.csv  ->", test_out.shape)

# %% [markdown]
# ## Summary
#
# - Dataset: IMDB, 25k train / 25k test, perfectly balanced (50/50 pos/neg)
# - Reviews range widely in length; a meaningful fraction exceed 256 words,
#   which will inform the `max_length` we choose for DistilBERT tokenization
# - Saved clean `text`/`label` CSVs to `data/` for the next steps
