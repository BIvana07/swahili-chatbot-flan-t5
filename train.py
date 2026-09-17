import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import torch

MODEL_NAME = "google/flan-t5-small"
CSV_PATH = "dataset.csv"

print("Loading CSV...")
df = pd.read_csv(CSV_PATH).dropna()
print(f"Rows: {len(df)}")

df = df.rename(columns={
    "swahili_input": "input",
    "swahili_output": "target"
})

train_df = df.sample(frac=0.8, random_state=42)
temp_df = df.drop(train_df.index)
val_df = temp_df.sample(frac=0.5, random_state=42)
test_df = temp_df.drop(val_df.index)

ds = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "val": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df),
})
print("Split sizes —")
print(ds)

print(f"Loading model & tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

MAX_LEN = 128

def preprocess(batch):
    inputs = tokenizer(
        batch["input"],
        max_length=MAX_LEN,
        truncation=True,
    )
    labels = tokenizer(
        batch["target"],
        max_length=MAX_LEN,
        truncation=True,
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

print("Tokenizing dataset...")
ds = ds.map(preprocess, batched=True)

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./flan_t5_swahili",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    push_to_hub=False,
    fp16=False,  # CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    tokenizer=tokenizer,
    data_collator=collator,
)

print("Starting training...")
trainer.train()

print("Saving final model...")
trainer.save_model("./flan_t5_swahili/final_model")
tokenizer.save_pretrained("./flan_t5_swahili/final_model")

print("Done.")
