# %% [markdown]
# # EDA: IMDB movie reviews
#
# Loading the dataset, checking class balance, and looking at review length
# before deciding how to tokenize later on.

# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset

sns.set_theme(style="whitegrid")

# %%
raw = load_dataset("imdb")
print(raw)

# %% [markdown]
# `train` and `test` are 25k reviews each; `unsupervised` is 50k unlabeled
# reviews we're not using. Each example is `text` + `label` (0 = negative,
# 1 = positive).

# %%
train_df = raw["train"].to_pandas()
test_df = raw["test"].to_pandas()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# %% [markdown]
# ## Class balance

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
# Balanced 50/50 in both splits, so accuracy is a fine metric here — no need
# for class weighting or resampling.
#
# ## A few raw examples

# %%
for i in range(3):
    label = "Positive" if train_df.iloc[i]["label"] == 1 else "Negative"
    print(f"--- Example {i} ({label}) ---")
    print(train_df.iloc[i]["text"][:300], "...\n")

# %% [markdown]
# Note the `<br />` tags — HTML line breaks left over from scraping. Worth
# cleaning up before training.
#
# ## Review length

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
for threshold in [128, 256, 512]:
    pct = (train_df["word_count"] > threshold).mean() * 100
    print(f"Reviews over {threshold} words: {pct:.1f}%")

# %% [markdown]
# DistilBERT caps out at 512 tokens, so this tells us roughly how much
# truncation to expect at different `max_length` settings — useful going
# into tokenization.
#
# ## Save splits

# %%
os.makedirs("../data", exist_ok=True)

train_out = train_df[["text", "label"]]
test_out = test_df[["text", "label"]]

train_out.to_csv("../data/train.csv", index=False)
test_out.to_csv("../data/test.csv", index=False)

print("Saved:")
print(" data/train.csv ->", train_out.shape)
print(" data/test.csv  ->", test_out.shape)
