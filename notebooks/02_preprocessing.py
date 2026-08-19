# %% [markdown]
# # Day 2 — Text Preprocessing
#
# On Day 1 we saved raw `text`/`label` splits to `data/train.csv` and
# `data/test.csv`. Today we prepare that text for DistilBERT.
#
# Key decision: DistilBERT's tokenizer (WordPiece) already lowercases,
# splits punctuation, and breaks words into subword pieces internally.
# Classic NLP steps — stemming, stopword removal, manual regex tokenizing —
# are deliberately NOT used here. They discard information (casing, word
# order, punctuation like "!"/"?") that transformer models actually use as
# signal, and empirically tend to hurt fine-tuning accuracy rather than
# help it.
#
# What we DO clean by hand is explained below: dataset-specific noise that
# isn't language at all.

# %%
import sys

sys.path.append("../src")

import pandas as pd

from preprocess import MAX_LENGTH, clean_text, get_tokenizer

# %% [markdown]
# ## 1. Load Day 1's saved splits

# %%
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
print("Train:", train_df.shape, " Test:", test_df.shape)

# %% [markdown]
# ## 2. Clean text: strip HTML artifacts
#
# IMDB reviews were scraped from a web page and contain literal `<br />`
# tags where line breaks were. These aren't part of the review's language —
# they're markup noise — so we strip them and collapse extra whitespace.
# Nothing else changes: casing, punctuation, and word order are left intact
# on purpose.

# %%
sample_raw = train_df.iloc[2]["text"]
print("--- BEFORE ---")
print(sample_raw[:300])
print("\n--- AFTER ---")
print(clean_text(sample_raw)[:300])

# %%
train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)

# Sanity check: no more <br /> tags left anywhere
assert not train_df["text"].str.contains("<br", case=False).any()
assert not test_df["text"].str.contains("<br", case=False).any()
print("HTML artifacts removed from all rows.")

# %% [markdown]
# ## 3. Tokenize with DistilBERT's tokenizer
#
# We use `distilbert-base-uncased`'s own WordPiece tokenizer. It:
# - Lowercases automatically ("uncased")
# - Splits into subword tokens, e.g. "unbelievable" -> ["un", "##believable"]
# - Adds special tokens: [CLS] at the start, [SEP] at the end
# - Pads/truncates every sequence to a fixed length so they batch into
#   tensors together
#
# From Day 1's EDA: 29.2% of reviews exceed 256 words, but the sentiment of
# a review is usually clear from its opening paragraphs. We use
# `max_length=256` as a speed/memory tradeoff for fine-tuning on a free
# Colab GPU (Day 3) — full 512-length coverage would train roughly 2x slower.

# %%
tokenizer = get_tokenizer()
print("Tokenizer max length setting used:", MAX_LENGTH)

# %%
example = clean_text(train_df.iloc[0]["text"])
encoded = tokenizer(example, truncation=True, max_length=MAX_LENGTH)

print("Original text (first 200 chars):", example[:200])
print("\nToken IDs (first 20):", encoded["input_ids"][:20])
print("\nDecoded back:", tokenizer.decode(encoded["input_ids"][:20]))
print("\nTotal tokens (post-truncation):", len(encoded["input_ids"]))

# %% [markdown]
# ## 4. Check truncation impact with the real tokenizer
#
# Day 1 estimated truncation using whitespace word counts. WordPiece
# subword tokens don't map 1:1 to words (e.g. one word can become 2-3
# tokens), so we recompute the real truncation rate here using the actual
# tokenizer — this is the number that matters for training.

# %%
sample = train_df["text"].sample(2000, random_state=42).tolist()
lengths = [len(tokenizer(t, truncation=False)["input_ids"]) for t in sample]
lengths = pd.Series(lengths)

print("Token length stats (sample of 2000 reviews):")
print(lengths.describe())
print(f"\n% of reviews truncated at max_length={MAX_LENGTH}:",
      f"{(lengths > MAX_LENGTH).mean() * 100:.1f}%")

# %% [markdown]
# ## 5. Save cleaned text (pre-tokenization) back to CSV
#
# We save the *cleaned text*, not tokenized tensors, to CSV — tokenized
# output depends on padding/batch size decisions that belong in the
# training script (Day 3), not baked into a static file. Keeping the saved
# artifact as clean text keeps this step reusable regardless of how Day 3
# decides to batch things.

# %%
train_df.to_csv("../data/train_clean.csv", index=False)
test_df.to_csv("../data/test_clean.csv", index=False)
print("Saved data/train_clean.csv and data/test_clean.csv")

# %% [markdown]
# ## Summary
#
# - Cleaned HTML artifacts (`<br />`) from all reviews; left casing,
#   punctuation, and word order untouched — over-cleaning hurts transformer
#   models
# - Confirmed DistilBERT's WordPiece tokenizer handles lowercasing and
#   subword splitting automatically
# - Recomputed real truncation rate at `max_length=256` using actual
#   subword token counts (not word counts)
# - Saved `data/train_clean.csv` / `data/test_clean.csv` for Day 3
#   (fine-tuning on Colab)
