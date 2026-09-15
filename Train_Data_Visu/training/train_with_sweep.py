from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import pandas as pd
import wandb
import os
import torch
import numpy as np
from typing import Dict, List
import re

def validate_spice_netlist(netlist: str) -> Dict[str, bool]:
    """Validate generated SPICE netlist"""
    lines = netlist.strip().split('\n')
    
    checks = {
        'has_source': False,
        'has_ground': False,
        'has_end': '.end' in netlist.lower(),
        'valid_format': True,
        'has_components': False
    }
    
    valid_prefixes = {'R', 'C', 'L', 'V', 'I', 'D', 'Q', 'M', 'X'}
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue
            
        parts = line.split()
        if not parts:
            continue
            
        prefix = parts[0][0].upper()
        
        if prefix not in valid_prefixes:
            checks['valid_format'] = False
            
        if prefix in ['R', 'C', 'L', 'D', 'Q', 'M']:
            checks['has_components'] = True
            
        if prefix in ['V', 'I']:
            checks['has_source'] = True
            
        if len(parts) >= 3 and '0' in [parts[1], parts[2]]:
            checks['has_ground'] = True
    
    return checks

class SPICEMetricsTrainer(Trainer):
    """Trainer extended with custom SPICE metrics"""
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        sample_size = min(10, len(eval_dataset))
        indices = np.random.choice(len(eval_dataset), sample_size, replace=False)
        
        valid_count = 0
        has_source_count = 0
        has_ground_count = 0
        has_end_count = 0
        has_components_count = 0
        
        print("\n" + "="*20)
        print(f"Generating {sample_size} examples for SPICE validation...")
        print("="*20)
        
        for idx in indices:
            input_text = eval_dataset[int(idx)]['input_text']
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=2.0
                )
            
            pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            checks = validate_spice_netlist(pred)
            
            if all(checks.values()):
                valid_count += 1
            if checks['has_source']:
                has_source_count += 1
            if checks['has_ground']:
                has_ground_count += 1
            if checks['has_end']:
                has_end_count += 1
            if checks['has_components']:
                has_components_count += 1
        
        spice_metrics = {
            f"{metric_key_prefix}/spice_valid_rate": valid_count / sample_size,
            f"{metric_key_prefix}/spice_has_source": has_source_count / sample_size,
            f"{metric_key_prefix}/spice_has_ground": has_ground_count / sample_size,
            f"{metric_key_prefix}/spice_has_end": has_end_count / sample_size,
            f"{metric_key_prefix}/spice_has_components": has_components_count / sample_size,
        }
        
        wandb.log(spice_metrics)
        
        output.update(spice_metrics)
        
        print("\nSPICE METRICS:")
        print(f"  Valid netlists: {valid_count}/{sample_size} ({100*valid_count/sample_size:.1f}%)")
        print(f"  With source: {has_source_count}/{sample_size} ({100*has_source_count/sample_size:.1f}%)")
        print(f"  With ground: {has_ground_count}/{sample_size} ({100*has_ground_count/sample_size:.1f}%)")
        print(f"  With .end: {has_end_count}/{sample_size} ({100*has_end_count/sample_size:.1f}%)")
        print(f"  With components: {has_components_count}/{sample_size} ({100*has_components_count/sample_size:.1f}%)")
        print("="*20 + "\n")
        
        return output

def train():
    """Training function compatible with wandb.agent"""
    
    run = wandb.init()
    config = wandb.config
    
    print("\n" + "="*20)
    print("STARTING NEW RUN")
    print("="*20)
    print(f"Run Name: {run.name}")
    print(f"Run ID: {run.id}")
    print("\nHYPERPARAMETERS:")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Batch Size: {config.per_device_train_batch_size}")
    print(f"  - Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Warmup Steps: {config.warmup_steps}")
    print(f"  - Weight Decay: {config.weight_decay}")
    print(f"  - Epochs: {config.num_train_epochs}")
    print(f"  - LR Scheduler: {config.lr_scheduler_type}")
    print(f"  - Effective Batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print("="*20 + "\n")
    
    print("Loading data...")
    data_files = {"train": ["results_augmented.csv"]}
    dataset = load_dataset("csv", data_files=data_files, delimiter=",", quoting=1)
    full_dataset = dataset["train"].shuffle(seed=42)
    
    print(f"Using full dataset ({len(full_dataset)} examples)")
    
    split_dataset = full_dataset.train_test_split(test_size=0.15)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"  Training: {len(train_dataset)} examples")
    print(f"  Validation: {len(eval_dataset)} examples\n")
    
    wandb.config.update({
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "total_samples": len(full_dataset)
    }, allow_val_change=True)
    
    print("Loading T5-small model...")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    wandb.config.update({
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params
    }, allow_val_change=True)
    
    def preprocess_function(examples):
        inputs = [str(ex) for ex in examples["input_text"]]
        targets = [str(ex) for ex in examples["output_text"]]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
        
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Preprocessing data...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True)
    print("  Tokenization completed\n")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = TrainingArguments(
        output_dir=f"./results/{run.name}",
        report_to="wandb",
        run_name=run.name,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f'./logs/{run.name}',
        logging_steps=20,
        logging_first_step=True,
        fp16=True,
        dataloader_num_workers=8,
        seed=42,
    )
    
    trainer = SPICEMetricsTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...\n")
    trainer.train()
    
    print("\nFinal evaluation...")
    final_metrics = trainer.evaluate()
    
    print("\nGenerating test examples...")
    
    test_prompts = [
        "A series circuit with 12V source, a 1k resistor, a 10uF capacitor and a 1mH inductor.",
        "A series circuit with 9V source, a 330 resistor and a diode.",
        "A simple RC low-pass filter powered by a 5V source.",
    ]
    
    examples_data = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=256,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=2.0
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        checks = validate_spice_netlist(generated)
        is_valid = "Valid" if all(checks.values()) else "Invalid"
        
        print(f"\n{is_valid} Input: {prompt}")
        print(f"   Output: {generated}")
        
        examples_data.append([prompt, generated, is_valid])
    
    examples_table = wandb.Table(
        columns=["Input", "Generated SPICE", "Valid"],
        data=examples_data
    )
    wandb.log({"examples": examples_table})
    
    if final_metrics.get("eval_loss", float('inf')) < 0.5:
        save_path = f"./best_models/{run.name}"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\nModel saved to {save_path}")
    
    print("\n" + "="*20)
    print("RUN COMPLETED")
    print("="*20 + "\n")
    
    wandb.finish()

if __name__ == "__main__":
    train()
