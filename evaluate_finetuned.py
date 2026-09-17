# evaluate_finetuned.py
# Usage: python evaluate_finetuned.py
# Assumes your fine-tuned model is saved in ./flan_t5_swahili/final_model
# and dataset.csv exists with columns swahili_input, swahili_output (or input,target).

import os
import csv
import statistics
from collections import Counter

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ----- CONFIG -----
LOCAL_MODEL_PATH = "./flan_t5_swahili/final_model"
CSV_PATH = "dataset.csv"
TEST_FRAC = 0.10
RANDOM_STATE = 42
MAX_GEN_LEN = 128
OUT_CSV = "predictions_finetuned_model.csv"
# -------------------

# simple token-F1 (bag/multiset overlap)
def token_f1_score(ref_tokens, pred_tokens):
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0, 0.0, 0.0
    ref_c = Counter(ref_tokens)
    pred_c = Counter(pred_tokens)
    common = sum(min(ref_c[t], pred_c[t]) for t in (set(ref_c) & set(pred_c)))
    precision = common / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = common / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# check local model folder
if not os.path.isdir(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"Local model folder not found: {LOCAL_MODEL_PATH}\n"
                            "If you saved the model to a different path, update LOCAL_MODEL_PATH.")

# load CSV
df = pd.read_csv(CSV_PATH).dropna()
# accept either naming convention
if "swahili_input" in df.columns and "swahili_output" in df.columns:
    df = df.rename(columns={"swahili_input": "input", "swahili_output": "target"})
if "input" not in df.columns or "target" not in df.columns:
    raise ValueError("CSV must have columns 'swahili_input'/'swahili_output' or 'input'/'target'")

# sample test set (same seed you used for training)
test_df = df.sample(frac=TEST_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)

# load tokenizer & model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading fine-tuned model from: {LOCAL_MODEL_PATH}  (device: {device})")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)
model.to(device)
model.eval()

smooth = SmoothingFunction().method1

bleu_scores = []
precisions = []
recalls = []
f1_scores = []
rows_out = []

print(f"Evaluating {len(test_df)} samples...")

for idx, row in test_df.iterrows():
    src = str(row["input"])
    tgt = str(row["target"])

    # generate
    try:
        enc = tokenizer(src, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_length=MAX_GEN_LEN)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    except Exception as e:
        pred = ""
        print(f"[Warning] generation failed for sample {idx}: {e}")

    # BLEU
    try:
        bleu = sentence_bleu([tgt.split()], pred.split(), smoothing_function=smooth)
    except Exception:
        bleu = 0.0

    # token-level metrics
    ref_tokens = tgt.split()
    pred_tokens = pred.split()
    p, r, f1 = token_f1_score(ref_tokens, pred_tokens)

    bleu_scores.append(bleu)
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f1)

    rows_out.append({
        "input": src,
        "target": tgt,
        "pred": pred,
        "bleu": bleu,
        "precision": p,
        "recall": r,
        "f1": f1
    })

    if (idx + 1) % 50 == 0:
        print(f"  processed {idx+1}/{len(test_df)}")

# averages
avg_bleu = statistics.mean(bleu_scores) if bleu_scores else 0.0
avg_p = statistics.mean(precisions) if precisions else 0.0
avg_r = statistics.mean(recalls) if recalls else 0.0
avg_f1 = statistics.mean(f1_scores) if f1_scores else 0.0

print("\n=== FINE-TUNED MODEL RESULTS ===")
print(f"Samples evaluated: {len(test_df)}")
print(f"Average BLEU:      {avg_bleu:.4f}")
print(f"Average Precision: {avg_p:.4f}")
print(f"Average Recall:    {avg_r:.4f}")
print(f"Average F1:        {avg_f1:.4f}")

# save CSV for report screenshots
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["input","target","pred","bleu","precision","recall","f1"])
    writer.writeheader()
    for r in rows_out:
        writer.writerow(r)

print(f"Saved per-sample predictions to: {OUT_CSV}")
