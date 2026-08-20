# %% [markdown]
# # Fine-tune DistilBERT on IMDB (run on Google Colab)
#
# Meant for Colab with a GPU runtime — fine-tuning on 25k examples takes
# hours on CPU, minutes on a free T4.
#
# Setup: upload this file plus `train_clean.csv`/`test_clean.csv` to Colab,
# run top to bottom, then download `model_output/` and drop it in this
# repo's `models/distilbert-sentiment/`.

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
# Should print "cuda" with a GPU runtime. If it prints "cpu": Runtime >
# Change runtime type > GPU.

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # matches src/preprocess.py

# %%
train_df = pd.read_csv("train_clean.csv")
test_df = pd.read_csv("test_clean.csv")
print("Train:", train_df.shape, " Test:", test_df.shape)

# For faster iteration/debugging, subsample first, e.g.:
# train_df = train_df.sample(2000, random_state=42).reset_index(drop=True)
# test_df = test_df.sample(1000, random_state=42).reset_index(drop=True)

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
# `num_labels=2` adds a fresh classification head on top of the pretrained
# base — that head starts randomly initialized and is what fine-tuning
# actually trains, while the base layers get nudged toward this task.

# %%
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)


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
    num_train_epochs=2,  # more starts to overfit on a dataset this size
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
# 2 epochs over 25k examples takes ~10-20 min on a T4.

# %%
trainer.train()

# %%
metrics = trainer.evaluate()
print(metrics)

# %% [markdown]
# Download `model_output/` afterward and place it at
# `models/distilbert-sentiment/` in the repo.

# %%
model.save_pretrained("model_output")
tokenizer.save_pretrained("model_output")
print("Saved to model_output/ — download this folder from Colab.")

# %%
# Optional: zip for easier download
# !zip -r model_output.zip model_output
