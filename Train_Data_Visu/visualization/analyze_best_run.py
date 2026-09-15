import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

PROJECT_NAME = "t5-spice-generator-base-ultimate"
BEST_RUN_NAME = "tough-sweep-2"

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("ANALYZING BEST RUN: " + BEST_RUN_NAME)
print("="*80)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

wandb.login()
api = wandb.Api()

print(f"\nConnecting to WandB project: {PROJECT_NAME}")

SWEEP_ID = "n01ywdhw"
sweep_path = f"{PROJECT_NAME}/{SWEEP_ID}"

print(f"Loading sweep: {SWEEP_ID}...")
sweep = api.sweep(sweep_path)
print(f"✓ Sweep loaded successfully")

print(f"\nSearching for run: {BEST_RUN_NAME}...")
run = None
for r in sweep.runs:
    if r.name == BEST_RUN_NAME:
        run = r
        break

if run is None:
    raise ValueError(f"Run '{BEST_RUN_NAME}' not found in sweep")

print(f"✓ Run found!")
print(f"\nDownloading training history (this may take a moment)...")
history = run.history(samples=100000, pandas=True)
print(f"✓ Downloaded {len(history)} data points")

history.to_csv(f"{OUTPUT_DIR}/{BEST_RUN_NAME}_full_history.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR}/{BEST_RUN_NAME}_full_history.csv")

config = run.config
summary = run.summary

train_loss_col = 'train/loss' if 'train/loss' in history.columns else None
eval_loss_col = 'eval/loss' if 'eval/loss' in history.columns else None
lr_col = 'train/learning_rate' if 'train/learning_rate' in history.columns else None
grad_norm_col = 'train/grad_norm' if 'train/grad_norm' in history.columns else None

spice_source_col = 'eval/spice_has_source' if 'eval/spice_has_source' in history.columns else None
spice_ground_col = 'eval/spice_has_ground' if 'eval/spice_has_ground' in history.columns else None
spice_end_col = 'eval/spice_has_end' if 'eval/spice_has_end' in history.columns else None

step_col = '_step' if '_step' in history.columns else history.index

print(f"\nGenerating figures...")
print(f"Output directory: {OUTPUT_DIR}")

# Training and Validation Loss
print("\n[1/4] Generating training and validation loss curves...")
fig, ax = plt.subplots(figsize=(14, 7))

if train_loss_col:
    train_data = history[[step_col, train_loss_col]].dropna()
    ax.plot(train_data[step_col], train_data[train_loss_col], 
           label='Training Loss', linewidth=2, color='#1f77b4', alpha=0.8)

if eval_loss_col:
    eval_data = history[[step_col, eval_loss_col]].dropna()
    ax.plot(eval_data[step_col], eval_data[eval_loss_col], 
           label='Validation Loss', linewidth=2.5, color='#ff7f0e', alpha=0.9)

ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
ax.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
ax.set_title(f'Training and Validation Loss Evolution - {BEST_RUN_NAME}', 
             fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_loss_curves.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/fig1_loss_curves.png")

if train_loss_col and eval_loss_col:
    train_final = history[train_loss_col].dropna().iloc[-1]
    eval_final = history[eval_loss_col].dropna().iloc[-1]
    train_initial = history[train_loss_col].dropna().iloc[0]
    eval_initial = history[eval_loss_col].dropna().iloc[0]

# Learning Rate Schedule
if lr_col:
    print("\n[2/4] Generating learning rate schedule...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    lr_data = history[[step_col, lr_col]].dropna()
    ax.plot(lr_data[step_col], lr_data[lr_col], 
           linewidth=2, color='#2ca02c', alpha=0.8)
    
    warmup_steps = config.get('warmup_steps', 0)
    if warmup_steps > 0:
        ax.axvline(warmup_steps, color='red', linestyle='--', linewidth=2, 
                  label=f'End of Warmup ({warmup_steps} steps)', alpha=0.7)
    
    ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title(f'Learning Rate Schedule ({config.get("lr_scheduler_type", "N/A")} scheduler)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_learning_rate.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig2_learning_rate.png")
    
    lr_max = lr_data[lr_col].max()
    lr_min = lr_data[lr_col].min()

# SPICE Metrics Evolution
if spice_source_col or spice_ground_col or spice_end_col:
    print("\n[3/4] Generating SPICE metrics evolution...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    if eval_loss_col:
        eval_data = history[[step_col, eval_loss_col]].dropna()
        ax1.plot(eval_data[step_col], eval_data[eval_loss_col], 
                linewidth=3, color='#ff7f0e', label='Validation Loss', alpha=0.9)
        ax1.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Validation Loss', fontsize=13, fontweight='bold', color='#ff7f0e')
        ax1.tick_params(axis='y', labelcolor='#ff7f0e')
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    
    if spice_source_col:
        source_data = history[[step_col, spice_source_col]].dropna()
        ax2.plot(source_data[step_col], source_data[spice_source_col] * 100, 
                linewidth=2, label='Has Source', marker='o', markersize=4, alpha=0.7)
    
    if spice_ground_col:
        ground_data = history[[step_col, spice_ground_col]].dropna()
        ax2.plot(ground_data[step_col], ground_data[spice_ground_col] * 100, 
                linewidth=2, label='Has Ground', marker='s', markersize=4, alpha=0.7)
    
    if spice_end_col:
        end_data = history[[step_col, spice_end_col]].dropna()
        ax2.plot(end_data[step_col], end_data[spice_end_col] * 100, 
                linewidth=2, label='Has .end', marker='^', markersize=4, alpha=0.7)
    
    ax2.set_ylabel('SPICE Validation Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='center right', fontsize=11)
    
    plt.title('SPICE Metrics Evolution During Training', 
             fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_spice_evolution.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig3_spice_evolution.png")

# Gradient Norm
if grad_norm_col:
    print("\n[4/4] Generating gradient norm plot...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    grad_data = history[[step_col, grad_norm_col]].dropna()
    
    window = 50
    grad_smoothed = grad_data[grad_norm_col].rolling(window=window, center=True).mean()
    
    ax.plot(grad_data[step_col], grad_data[grad_norm_col], 
           linewidth=0.5, color='lightblue', alpha=0.3, label='Raw Gradient Norm')
    ax.plot(grad_data[step_col], grad_smoothed, 
           linewidth=2, color='#d62728', label=f'Moving Average (window={window})')
    
    ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=13, fontweight='bold')
    ax.set_title('Gradient Norm During Training (Stability Indicator)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_gradient_norm.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig4_gradient_norm.png")
    
    grad_mean = grad_data[grad_norm_col].mean()
    grad_std = grad_data[grad_norm_col].std()
    grad_max = grad_data[grad_norm_col].max()

# Summary statistics
with open(f"{OUTPUT_DIR}/report_statistics.txt", "w", encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("STATISTICS FOR SECTION 5.2 - TRAINING DYNAMICS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"RUN: {BEST_RUN_NAME}\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Learning Rate: {config.get('learning_rate', 'N/A')}\n")
    f.write(f"  Batch Size: {config.get('per_device_train_batch_size', 'N/A')}\n")
    f.write(f"  Gradient Accumulation: {config.get('gradient_accumulation_steps', 'N/A')}\n")
    f.write(f"  Effective Batch Size: {config.get('per_device_train_batch_size', 1) * config.get('gradient_accumulation_steps', 1)}\n")
    f.write(f"  Warmup Steps: {config.get('warmup_steps', 'N/A')}\n")
    f.write(f"  Total Epochs: {config.get('num_train_epochs', 'N/A')}\n")
    f.write(f"  Scheduler: {config.get('lr_scheduler_type', 'N/A')}\n\n")
    
    if train_loss_col and eval_loss_col:
        f.write("CONVERGENCE BEHAVIOR:\n")
        f.write(f"  Initial Training Loss: {train_initial:.4f}\n")
        f.write(f"  Final Training Loss: {train_final:.4f}\n")
        f.write(f"  Training Loss Reduction: {(1-train_final/train_initial)*100:.1f}%\n")
        f.write(f"  Initial Validation Loss: {eval_initial:.4f}\n")
        f.write(f"  Final Validation Loss: {eval_final:.4f}\n")
        f.write(f"  Validation Loss Reduction: {(1-eval_final/eval_initial)*100:.1f}%\n")
        f.write(f"  Train-Val Gap: {abs(train_final - eval_final):.4f}\n\n")
    
    if lr_col:
        f.write("LEARNING RATE SCHEDULE:\n")
        f.write(f"  Peak LR (after warmup): {lr_max:.2e}\n")
        f.write(f"  Final LR: {lr_min:.2e}\n")
        f.write(f"  Decay Ratio: {lr_min/lr_max:.4f}\n\n")
    
    if grad_norm_col:
        f.write("TRAINING STABILITY (Gradient Norm):\n")
        f.write(f"  Mean: {grad_mean:.4f}\n")
        f.write(f"  Std Dev: {grad_std:.4f}\n")
        f.write(f"  Max: {grad_max:.4f}\n\n")
    
    f.write("SPICE METRICS (FINAL):\n")
    f.write(f"  Has Source: {summary.get('eval/spice_has_source', 0)*100:.1f}%\n")
    f.write(f"  Has Ground: {summary.get('eval/spice_has_ground', 0)*100:.1f}%\n")
    f.write(f"  Has .end: {summary.get('eval/spice_has_end', 0)*100:.1f}%\n\n")
    
    f.write("TRAINING METRICS:\n")
    f.write(f"  Total Steps: {summary.get('train/global_step', 'N/A')}\n")
    f.write(f"  Total Epochs: {summary.get('train/epoch', 'N/A')}\n")
    f.write(f"  Training Time: {summary.get('_runtime', 'N/A')} seconds\n")

print(f"\n✓ Saved: {OUTPUT_DIR}/report_statistics.txt")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print(f"  - {BEST_RUN_NAME}_full_history.csv")
print(f"  - fig1_loss_curves.png")
if lr_col:
    print(f"  - fig2_learning_rate.png")
if spice_source_col or spice_ground_col or spice_end_col:
    print(f"  - fig3_spice_evolution.png")
if grad_norm_col:
    print(f"  - fig4_gradient_norm.png")
print(f"  - report_statistics.txt")
