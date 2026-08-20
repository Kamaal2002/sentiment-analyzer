# %% [markdown]
# # Interpretability: why does the model predict what it predicts?
#
# DistilBERT beats the TF-IDF baseline by 5% relative F1, but that number
# alone doesn't say *why* — genuine use of context and tone, or just a
# bigger vocabulary of keyword correlations? Captum's **Layer Integrated
# Gradients** attributes each prediction back to individual input tokens,
# so we can see which words actually push the model toward "positive" or
# "negative."
#
# It works by comparing the model's output on the real input against a
# neutral baseline input (here, all-[PAD] tokens) and attributing the
# difference back to each token based on how much moving from baseline to
# real input changed the prediction — a standard, well-established method,
# not something specific to this project.
#
# Three cases worth checking: confident correct predictions (sanity check),
# DistilBERT-correct/baseline-wrong (what extra signal is it using?), and
# both-models-wrong ("right for the wrong reasons" or genuinely hard?).

# %%
import sys

sys.path.append("../src")

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

DEVICE = "cpu"  # interpretability runs on single examples; CPU is fine here

# %%
model = DistilBertForSequenceClassification.from_pretrained("../models/distilbert-sentiment")
tokenizer = DistilBertTokenizerFast.from_pretrained("../models/distilbert-sentiment")
model.to(DEVICE)
model.eval()

LABELS = {0: "Negative", 1: "Positive"}

# %% [markdown]
# Attributing with respect to the input embedding layer is the standard
# setup for transformer attribution — it's where tokens still correspond
# 1:1 to positions, before attention mixes them together.


# %%
def predict_forward(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


lig = LayerIntegratedGradients(predict_forward, model.distilbert.embeddings)


def explain(text, target_class=None, n_steps=50):
    """Return (tokens, attributions, predicted_class, confidence) for one text."""
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = predict_forward(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_class = torch.argmax(logits, dim=-1).item()

    if target_class is None:
        target_class = pred_class

    # Baseline: same sequence but every token replaced with [PAD], keeping
    # [CLS]/[SEP] in place — the standard "neutral input" reference point.
    baseline_ids = input_ids.clone()
    baseline_ids[:, 1:-1] = tokenizer.pad_token_id

    attributions, _ = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    # Sum over the embedding dimension to get one score per token
    token_attributions = attributions.sum(dim=-1).squeeze(0)
    token_attributions = token_attributions / torch.norm(token_attributions)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, token_attributions.detach().numpy(), pred_class, probs[pred_class].item()


def print_attribution(text, target_class=None):
    tokens, attrs, pred_class, conf = explain(text, target_class)
    print(f"Prediction: {LABELS[pred_class]} (confidence: {conf:.3f})\n")
    for tok, score in zip(tokens, attrs):
        marker = "+" if score > 0 else "-"
        bar = "#" * min(int(abs(score) * 20), 20)
        print(f"  {marker} {tok:<15} {score:+.3f} {bar}")


# %% [markdown]
# ## Sanity check: clear-cut examples
#
# Should see strong positive attribution on obviously positive words and
# vice versa — if not, something's wrong with the setup before trusting it
# on harder cases.

# %%
print_attribution("This movie was absolutely fantastic, the acting was incredible.")

# %%
print_attribution("Terrible film, a complete waste of time.")

# %% [markdown]
# ## Cases where DistilBERT succeeded but the baseline failed
#
# Pulled from the earlier model comparison. If DistilBERT is genuinely using
# context and not just a bigger keyword vocabulary, attributions here should
# highlight negation, contrast, or tone rather than isolated sentiment words.

# %%
distilbert_wins_examples = [
    "Ninja Hunter (AKA Wu Tang vs Ninja) is pure entertainment from start "
    "to finish due to its outrageous characters, nonsensical plot and lack "
    "of any pretensions whatsoever.",
    "I don't really know why so many persons love this movie: maybe it's "
    "funny, OK, but it has totally ruined one of the best novels ever "
    "written.",
]

for text in distilbert_wins_examples:
    print_attribution(text)
    print("\n" + "-" * 70 + "\n")

# %% [markdown]
# ## "Right for the wrong reasons" check
#
# A model can get the right label while attending to something irrelevant
# (an actor's name, a genre keyword) instead of actual sentiment language.
# Checking a batch of correct high-confidence predictions for cases where
# the top-attributed tokens aren't recognizable sentiment words — a real
# failure mode even when the final label is correct.

# %%
test_df = pd.read_csv("../data/test_clean.csv").sample(20, random_state=7).reset_index(drop=True)

for _, row in test_df.iterrows():
    tokens, attrs, pred_class, conf = explain(row["text"][:500])
    if pred_class != row["label"] or conf < 0.9:
        continue
    top_idx = attrs.argsort()[-3:][::-1]
    top_tokens = [tokens[i] for i in top_idx if tokens[i] not in ("[CLS]", "[SEP]")]
    print(f"Pred: {LABELS[pred_class]} ({conf:.2f})  Top tokens: {top_tokens}")

# %% [markdown]
# ## Both models wrong: genuinely hard cases
#
# Examples both the baseline and DistilBERT misclassified. Attribution
# tells us whether the model was at least "reasoning" sensibly and got
# unlucky (e.g. sarcasm it couldn't detect), versus latching onto something
# spurious.

# %%
both_wrong_examples = [
    ("Sometimes a movie is so bad it's kind of good. This movie was made "
     "in Germany and is dubbed in English, so you have to get past that. "
     "The acting also was stilted and forced, other than what was done by "
     "the real rats, who IMO did an excellent job of acting the part.", 1),
    ("'Ernest Saves Christmas' is comedian Ernest's Christmas special "
     "film. In this film, Ernest has to find a successor to Santa Claus "
     "in order for Christmas to continue.", 0),
]

for text, true_label in both_wrong_examples:
    print(f"True label: {LABELS[true_label]}")
    print_attribution(text)
    print("\n" + "-" * 70 + "\n")

# %% [markdown]
# ## Save example attributions
#
# Saved as JSON so the same explain() output format can be reused by the
# Flask API to serve token-level attributions alongside a prediction, and
# by the README for a concrete before/after example.

# %%
import json

case_studies = {
    "sanity_check_positive": "This movie was absolutely fantastic, the acting was incredible.",
    "sanity_check_negative": "Terrible film, a complete waste of time.",
    "distilbert_wins_context": distilbert_wins_examples[1],
    "sarcasm_failure": both_wrong_examples[0][0],
}

saved = {}
for name, text in case_studies.items():
    tokens, attrs, pred_class, conf = explain(text)
    saved[name] = {
        "text": text,
        "prediction": LABELS[pred_class],
        "confidence": round(conf, 4),
        "tokens": tokens,
        "attributions": [round(float(a), 4) for a in attrs],
    }

with open("../data/interpretability_examples.json", "w") as f:
    json.dump(saved, f, indent=2)

print("Saved data/interpretability_examples.json")
