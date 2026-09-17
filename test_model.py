from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_PATH = "flan_t5_swahili/final_model"

print(f"Loading model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = torch.device("cpu")
model.to(device)

print("Ready! Type Swahili text... (exit to quit)")

while True:
    text = input("You: ").strip()
    if text.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(text, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    print("Bot:", tokenizer.decode(output[0], skip_special_tokens=True))
