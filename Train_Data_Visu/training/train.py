from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import pandas as pd
import wandb
import os

OUTPUT_DIR_DRIVE = "spice_model_final" 
PROJECT_NAME = "t5-spice-generator"

print("Loading data...")
data_files = {"train": ["results_augmented.csv"]}

dataset = load_dataset("csv", data_files=data_files, delimiter=",", quoting=1)
full_dataset = dataset["train"].shuffle(seed=42)

split_dataset = full_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training on {len(train_dataset)} examples.")
print(f"Validation on {len(eval_dataset)} examples.")

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["output_text"]]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    run_name="run_t5_colab_improved",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=3e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=50,
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir='./logs',
    logging_steps=100,
    fp16=True,
    warmup_steps=1000,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print(f"Saving final model to {OUTPUT_DIR_DRIVE}...")
model.save_pretrained(OUTPUT_DIR_DRIVE)
tokenizer.save_pretrained(OUTPUT_DIR_DRIVE)

print("Training completed.")
wandb.finish()
