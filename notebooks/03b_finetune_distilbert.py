# %% [markdown]
# # Day 3b — Fine-tune DistilBERT on IMDB (run this on Google Colab)
#
# This notebook is meant to run on Colab with a free GPU (Runtime > Change
# runtime type > GPU). Fine-tuning a transformer on 25k examples is slow on
# CPU (potentially hours) but fast on a GPU (a few minutes per epoch).
#
# ## Setup on Colab
#
# 1. Upload this file to Colab (or open directly from GitHub once pushed).
# 2. Upload `data/train_clean.csv` and `data/test_clean.csv` from this repo
#    (Colab's file browser on the left, or mount Google Drive).
# 3. Run the cells top to bottom.
# 4. At the end, download the `model_output/` folder (zipped) and place it
#    in this repo's `models/distilbert-sentiment/` directory.

# %%
# On Colab, uncomment and run this first:
# !pip install transformers datasets torch scikit-learn -q

# %%
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
# On Colab with GPU runtime enabled, this should print "cuda".
# If it prints "cpu", go to Runtime > Change runtime type > GPU.

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # matches src/preprocess.py from Day 2

# %% [markdown]
# ## 1. Load cleaned data
#
# Update these paths if you uploaded the CSVs somewhere else on Colab.

# %%
train_df = pd.read_csv("train_clean.csv")
test_df = pd.read_csv("test_clean.csv")
print("Train:", train_df.shape, " Test:", test_df.shape)

# For faster iteration/debugging, you can subsample first, e.g.:
# train_df = train_df.sample(2000, random_state=42).reset_index(drop=True)
# test_df = test_df.sample(1000, random_state=42).reset_index(drop=True)

# %% [markdown]
# ## 2. Tokenize
#
# Same tokenizer, same max_length as Day 2's preprocessing — keeping this
# consistent matters so the model is trained on the same representation
# we'll use at inference time in the API.

# %%
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=MAX_LENGTH
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = IMDBDataset(train_df["text"], train_df["label"])
test_dataset = IMDBDataset(test_df["text"], test_df["label"])

# %% [markdown]
# ## 3. Load pretrained DistilBERT with a classification head
#
# `distilbert-base-uncased` is pretrained on general English text (masked
# language modeling), not on sentiment. `num_labels=2` adds a fresh linear
# classification head on top — that head starts randomly initialized and
# is what fine-tuning actually trains, while the pretrained base layers get
# nudged to be useful for this specific task.

# %%
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)


# %% [markdown]
# ## 4. Training setup

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,  # 2 epochs is usually enough for IMDB; more risks overfitting
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## 5. Train
#
# On a Colab T4 GPU, 2 epochs over 25k examples typically takes ~10-20
# minutes. Watch the eval metrics printed after each epoch.

# %%
trainer.train()

# %% [markdown]
# ## 6. Final evaluation

# %%
metrics = trainer.evaluate()
print(metrics)

# %% [markdown]
# ## 7. Save the fine-tuned model
#
# Download the `model_output` folder afterward (right-click in Colab's file
# browser > Download, or zip it first) and place it at
# `models/distilbert-sentiment/` in the repo.

# %%
model.save_pretrained("model_output")
tokenizer.save_pretrained("model_output")
print("Saved to model_output/ — download this folder from Colab.")

# %%
# Optional: zip for easier download
# !zip -r model_output.zip model_output
