# %% [markdown]
# # Preprocessing for DistilBERT
#
# DistilBERT's tokenizer already lowercases, splits punctuation, and breaks
# words into subwords. Classic NLP preprocessing (stemming, stopword
# removal, manual tokenizing) throws away information transformers actually
# use — casing, word order, punctuation — and tends to hurt fine-tuning
# accuracy rather than help it. So the only manual cleaning here is
# stripping the `<br />` HTML artifacts noticed during EDA; everything else
# is left for the tokenizer to handle.

# %%
import sys

sys.path.append("../src")

import pandas as pd

from preprocess import MAX_LENGTH, clean_text, get_tokenizer

# %%
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
print("Train:", train_df.shape, " Test:", test_df.shape)

# %% [markdown]
# ## Clean text

# %%
sample_raw = train_df.iloc[2]["text"]
print("--- BEFORE ---")
print(sample_raw[:300])
print("\n--- AFTER ---")
print(clean_text(sample_raw)[:300])

# %%
train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)

assert not train_df["text"].str.contains("<br", case=False).any()
assert not test_df["text"].str.contains("<br", case=False).any()
print("HTML artifacts removed from all rows.")

# %% [markdown]
# ## Tokenize
#
# `distilbert-base-uncased` lowercases automatically and splits into
# subwords (e.g. "unbelievable" -> `["un", "##believable"]`), then adds
# `[CLS]`/`[SEP]` and pads/truncates to a fixed length.

# %%
tokenizer = get_tokenizer()
print("max_length:", MAX_LENGTH)

# %%
example = clean_text(train_df.iloc[0]["text"])
encoded = tokenizer(example, truncation=True, max_length=MAX_LENGTH)

print("Original text (first 200 chars):", example[:200])
print("\nToken IDs (first 20):", encoded["input_ids"][:20])
print("\nDecoded back:", tokenizer.decode(encoded["input_ids"][:20]))
print("\nTotal tokens (post-truncation):", len(encoded["input_ids"]))

# %% [markdown]
# The EDA's truncation estimate was based on whitespace word counts, but
# subword tokens don't map 1:1 to words — one word can become 2-3 tokens.
# Worth recomputing the real truncation rate with the actual tokenizer.

# %%
sample = train_df["text"].sample(2000, random_state=42).tolist()
lengths = [len(tokenizer(t, truncation=False)["input_ids"]) for t in sample]
lengths = pd.Series(lengths)

print("Token length stats (sample of 2000 reviews):")
print(lengths.describe())
print(f"\n% of reviews truncated at max_length={MAX_LENGTH}:",
      f"{(lengths > MAX_LENGTH).mean() * 100:.1f}%")

# %% [markdown]
# That's higher than the word-count estimate suggested (~40% vs ~29%), as
# expected. Keeping `max_length=256` anyway — it's a reasonable
# speed/memory tradeoff for fine-tuning on a free Colab GPU, and sentiment
# is usually clear from the first chunk of a review.
#
# ## Save cleaned splits
#
# Saving cleaned text rather than tokenized tensors — tokenization details
# like padding/batch size are a training-time decision, not something to
# bake into the saved file.

# %%
train_df.to_csv("../data/train_clean.csv", index=False)
test_df.to_csv("../data/test_clean.csv", index=False)
print("Saved data/train_clean.csv and data/test_clean.csv")
