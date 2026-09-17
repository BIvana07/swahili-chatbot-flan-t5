# evaluate_base_fixed.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import csv

# --- CONFIG ---
MODEL_NAME = "google/flan-t5-small"     # change to "./flan_t5_swahili/final_model" for fine-tuned eval
CSV_PATH = "dataset.csv"
TEST_FRAC = 0.10           # fraction to use as test set (same seed used for both models)
RANDOM_STATE = 42
MAX_GEN_LENGTH = 128

# --- helpers ---
smooth = SmoothingFunction().method1

def token_f1_score(ref_tokens, pred_tokens):
    # compute token-level precision, recall, f1 for one sample
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0:
        return 0.0, 0.0, 0.0
    if len(ref_tokens) == 0:
        return 0.0, 0.0, 0.0

    # count common tokens (multiset / bag intersection)
    from collections import Counter
    ref_c = Counter(ref_tokens)
    pred_c = Counter(pred_tokens)
    common = sum(min(ref_c[t], pred_c[t]) for t in (set(ref_c) & set(pred_c)))
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

# --- load data ---
df = pd.read_csv(CSV_PATH).dropna()
# ensure column names match
if "swahili_input" in df.columns and "swahili_output" in df.columns:
    df = df.rename(columns={"swahili_input": "input", "swahili_output": "target"})
elif "input" not in df.columns or "target" not in df.columns:
    raise ValueError("CSV must have columns 'swahili_input'/'swahili_output' or 'input'/'target'")

test_df = df.sample(frac=TEST_FRAC, random_state=RANDOM_STATE)

# --- load model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- evaluate ---
bleu_scores = []
f1_scores = []
precisions = []
recalls = []

rows_out = []

print(f"Evaluating model: {MODEL_NAME} on {len(test_df)} samples ...")

for _, row in test_df.iterrows():
    src = str(row["input"])
    tgt = str(row["target"])

    # tokenise and generate
    inputs = tokenizer(src, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outs = model.generate(**inputs, max_length=MAX_GEN_LENGTH)
    pred = tokenizer.decode(outs[0], skip_special_tokens=True).strip()

    # BLEU (sentence-level)
    try:
        bleu = sentence_bleu([tgt.split()], pred.split(), smoothing_function=smooth)
    except Exception:
        bleu = 0.0

    # token-level F1
    ref_tokens = tgt.split()
    pred_tokens = pred.split()
    p, r, f1 = token_f1_score(ref_tokens, pred_tokens)

    bleu_scores.append(bleu)
    f1_scores.append(f1)
    precisions.append(p)
    recalls.append(r)

    rows_out.append({
        "input": src,
        "target": tgt,
        "pred": pred,
        "bleu": bleu,
        "precision": p,
        "recall": r,
        "f1": f1
    })

# --- results ---
import statistics
avg_bleu = statistics.mean(bleu_scores) if bleu_scores else 0.0
avg_f1 = statistics.mean(f1_scores) if f1_scores else 0.0
avg_p = statistics.mean(precisions) if precisions else 0.0
avg_r = statistics.mean(recalls) if recalls else 0.0

print("=== RESULTS ===")
print(f"Samples evaluated: {len(test_df)}")
print(f"Average BLEU:      {avg_bleu:.4f}")
print(f"Average Precision: {avg_p:.4f}")
print(f"Average Recall:    {avg_r:.4f}")
print(f"Average F1:        {avg_f1:.4f}")

# --- save CSV of predictions for report screenshots ---
out_csv = f"predictions_{MODEL_NAME.replace('/', '_').replace('.', '_')}.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["input","target","pred","bleu","precision","recall","f1"])
    writer.writeheader()
    for r in rows_out:
        writer.writerow(r)

print(f"Saved per-sample predictions to: {out_csv}")
