import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

PROJECT_NAME = "t5-spice-generator-base-ultimate"
SWEEP_ID = "n01ywdhw"

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("ANALYZING SWEEP RESULTS")
print("="*80)
print(f"Project: {PROJECT_NAME}")
print(f"Sweep ID: {SWEEP_ID}")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

wandb.login()
api = wandb.Api()

print("\nConnecting to WandB...")
sweep_path = f"{PROJECT_NAME}/{SWEEP_ID}"
print(f"Loading sweep: {sweep_path}...")
sweep = api.sweep(sweep_path)
print("✓ Sweep loaded successfully")

print("\nExtracting run data...")
runs_data = []

for run in sweep.runs:
    run_info = {
        'name': run.name,
        'id': run.id,
        'state': run.state,
        'created_at': run.created_at,
        'runtime_seconds': run.summary.get('_runtime', 0),
    }
    
    def get_metric(summary, *keys):
        for key in keys:
            if key in summary:
                return summary[key]
        return None
    
    run_info.update({
        'eval_loss': get_metric(run.summary, 'eval/loss', 'eval_loss'),
        'train_loss': get_metric(run.summary, 'train/loss', 'train_loss'),
        'spice_has_source': get_metric(run.summary, 'eval/spice_has_source', 'spice_has_source', 'eval_spice_has_source'),
        'spice_has_ground': get_metric(run.summary, 'eval/spice_has_ground', 'spice_has_ground', 'eval_spice_has_ground'),
        'spice_has_end': get_metric(run.summary, 'eval/spice_has_end', 'spice_has_end', 'eval_spice_has_end'),
    })
    
    config = run.config
    run_info.update({
        'learning_rate': config.get('learning_rate'),
        'batch_size': config.get('per_device_train_batch_size'),
        'grad_accum': config.get('gradient_accumulation_steps'),
        'warmup_steps': config.get('warmup_steps'),
        'weight_decay': config.get('weight_decay'),
        'epochs': config.get('num_train_epochs'),
        'scheduler': config.get('lr_scheduler_type'),
    })
    
    if run_info['batch_size'] and run_info['grad_accum']:
        run_info['effective_batch'] = run_info['batch_size'] * run_info['grad_accum']
    
    runs_data.append(run_info)

df = pd.DataFrame(runs_data)
df = df.sort_values('created_at').reset_index(drop=True)
df['run_number'] = range(1, len(df) + 1)

print(f"✓ Extracted data from {len(df)} runs")
print(f"  - Completed: {len(df[df['state'] == 'finished'])}")
print(f"  - With metrics: {len(df[df['eval_loss'].notna()])}")

df_top5 = df[df['eval_loss'].notna()].nsmallest(5, 'eval_loss')
if len(df_top5) > 0:
    print(f"  - Best validation loss: {df_top5.iloc[0]['eval_loss']:.4f}")

print("\nGenerating figures...")
print(f"Output directory: {OUTPUT_DIR}")

# Figure 1: Sweep Progress
print("\n[1/8] Generating sweep progress plot...")
fig, ax = plt.subplots(figsize=(12, 6))

df_valid = df[df['eval_loss'].notna()].copy()

successful_runs = df_valid[df_valid['eval_loss'] < 1.0].copy()
failed_runs = df_valid[df_valid['eval_loss'] >= 1.0].copy()

successful_runs['best_loss_so_far'] = successful_runs['eval_loss'].cummin()

ax.plot(successful_runs['run_number'], successful_runs['eval_loss'], 
        'o', alpha=0.5, markersize=10, label='Individual runs', color='lightblue')
ax.plot(successful_runs['run_number'], successful_runs['best_loss_so_far'], 
        '-o', linewidth=3, markersize=8, label='Best so far', color='darkblue')

if len(failed_runs) > 0:
    ax.plot(failed_runs['run_number'], failed_runs['eval_loss'], 
            'x', markersize=12, label='Failed runs (excluded)', 
            color='red', markeredgewidth=2)

ax.set_xlabel('Run Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax.set_title('Sweep Progress: Best Validation Loss vs. Run Number', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

if len(successful_runs) > 0:
    min_loss = successful_runs['eval_loss'].min()
    max_loss = successful_runs['eval_loss'].max()
    margin = (max_loss - min_loss) * 0.05
    
    if margin < 0.002:
        margin = 0.002
    
    ax.set_ylim(min_loss - margin, max_loss + margin)
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_sweep_progress.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/fig1_sweep_progress.png")

# Figure 2: Parallel Coordinates Plot
print("\n[2/8] Generating parallel coordinates plot...")
hyperparams_pc = ['learning_rate', 'batch_size', 'grad_accum', 'warmup_steps', 
                  'weight_decay', 'epochs', 'eval_loss']

df_pc = df[hyperparams_pc + ['name']].dropna()

if len(df_pc) > 0:
    df_pc['is_top5'] = df_pc['name'].isin(df_top5['name']) if len(df_top5) > 0 else False
    
    df_normalized = df_pc.copy()
    for col in hyperparams_pc:
        if col == 'learning_rate' or col == 'weight_decay':
            df_normalized[col] = np.log10(df_pc[col])
        
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val > min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0.5
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for idx, row in df_normalized.iterrows():
        if not row['is_top5']:
            values = row[hyperparams_pc].values
            ax.plot(range(len(hyperparams_pc)), values, 
                   color='lightgray', alpha=0.3, linewidth=1)
    
    colors_top5 = ['#00FF00', '#FFD700', '#FF6B6B', '#4ECDC4', '#95E1D3']
    for i, (idx, row) in enumerate(df_normalized[df_normalized['is_top5']].iterrows()):
        values = row[hyperparams_pc].values
        color = colors_top5[i] if i < len(colors_top5) else 'blue'
        ax.plot(range(len(hyperparams_pc)), values, 
               color=color, alpha=0.9, linewidth=3, 
               label=f"{df_pc.loc[idx, 'name']} (loss: {df_pc.loc[idx, 'eval_loss']:.4f})")
    
    ax.set_xticks(range(len(hyperparams_pc)))
    ax.set_xticklabels([p.replace('_', ' ').title() for p in hyperparams_pc], 
                       rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Coordinates Plot: Hyperparameters and Validation Loss', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    for x in range(len(hyperparams_pc)):
        ax.axvline(x, color='gray', alpha=0.2, linestyle='--')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, 
             title='Top 5 Runs', title_fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_parallel_coordinates.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig2_parallel_coordinates.png")

# Figure 3: Hyperparameter Importance
print("\n[3/8] Generating hyperparameter importance plot...")
hyperparams = ['learning_rate', 'batch_size', 'grad_accum', 'warmup_steps', 
               'weight_decay', 'epochs', 'effective_batch']

df_corr = df[hyperparams + ['eval_loss']].dropna()

if len(df_corr) > 3:
    valid_params = []
    for param in hyperparams:
        if df_corr[param].std() > 1e-10:
            valid_params.append(param)
    
    if len(valid_params) > 0:
        correlations = df_corr[valid_params].corrwith(df_corr['eval_loss']).abs()
        correlations = correlations.sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['cyan' if x > 0 else 'pink' for x in df_corr[valid_params].corrwith(df_corr['eval_loss'])]
        correlations.plot(kind='barh', ax=ax, color=colors)
        
        ax.set_xlabel('Correlation with Validation Loss (absolute)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hyperparameter', fontsize=12, fontweight='bold')
        ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig3_hyperparameter_importance.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_DIR}/fig3_hyperparameter_importance.png")

# Figure 4: SPICE Metrics Breakdown
print("\n[4/8] Generating SPICE metrics breakdown...")
df_spice_data = df[df['eval_loss'].notna()].copy()

if len(df_spice_data) > 0:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    spice_cols = ['spice_has_source', 'spice_has_ground', 'spice_has_end']
    df_spice_all = df_spice_data[['name', 'eval_loss'] + spice_cols].copy()
    
    x_pos = np.arange(len(df_spice_all))
    width = 0.25
    
    for i, col in enumerate(spice_cols):
        values = df_spice_all[col].fillna(0) * 100
        ax.bar(x_pos + i*width, values, width, 
               label=col.replace('spice_has_', '').replace('_', ' ').title(),
               alpha=0.8)
    
    ax.set_xlabel('Run Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('SPICE Validation Metrics by Run', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(df_spice_all['name'], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_spice_metrics.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig4_spice_metrics.png")

# Figure 5: SPICE Metrics Average
print("\n[5/8] Generating SPICE metrics average...")
spice_metrics = ['spice_has_source', 'spice_has_ground', 'spice_has_end']

df_spice = df[spice_metrics].dropna()

if len(df_spice) > 0:
    means = df_spice.mean() * 100
    stds = df_spice.std() * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(spice_metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    
    ax.set_xlabel('SPICE Validator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('SPICE Validation Metrics (Average across all runs)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Has Source', 'Has Ground', 'Has .end'], 
                       rotation=0, ha='center')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_spice_metrics_avg.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig5_spice_metrics_avg.png")

# Figure 6: Top 5 Runs Comparison
print("\n[6/8] Generating top 5 runs comparison...")
if len(df_top5) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    params_to_plot = ['learning_rate', 'effective_batch', 'warmup_steps', 
                      'weight_decay', 'epochs', 'eval_loss']
    
    for i, param in enumerate(params_to_plot):
        ax = axes[i]
        values = df_top5[param].values
        names = [f"Run {i+1}" for i in range(len(df_top5))]
        
        bars = ax.bar(names, values, alpha=0.7, edgecolor='black')
        ax.set_ylabel(param.replace('_', ' ').title(), fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        bars[0].set_color('lightgreen')
        bars[0].set_edgecolor('darkgreen')
        bars[0].set_linewidth(2)
    
    plt.suptitle('Top 5 Runs: Hyperparameter Comparison', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_top5_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig6_top5_comparison.png")

# Top 5 Table
if len(df_top5) > 0:
    table_cols = ['name', 'eval_loss', 'learning_rate', 
                  'effective_batch', 'scheduler', 'epochs']
    
    df_table = df_top5[table_cols].copy()
    df_table['eval_loss'] = df_table['eval_loss'].apply(
        lambda x: round(x, 4) if pd.notna(x) else None
    )
    
    df_table.to_csv(f"{OUTPUT_DIR}/table_top5_configs.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR}/table_top5_configs.csv")

# Figure 7: Training Curves (Best Run)
print("\n[7/8] Generating training curves for best run...")
best_run_name = df_top5.iloc[0]['name']
best_run_id = df_top5.iloc[0]['id']

best_run = None
for run in sweep.runs:
    if run.id == best_run_id:
        best_run = run
        break

if best_run is not None:
    history = best_run.history(samples=10000, pandas=True)
    
    if history is not None and len(history) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        has_train = 'train/loss' in history.columns
        has_eval = 'eval/loss' in history.columns
        
        if has_train or has_eval:
            if has_train:
                train_data = history[['_step', 'train/loss']].dropna()
                if len(train_data) > 0:
                    ax.plot(train_data['_step'], train_data['train/loss'], 
                            label='Training Loss', linewidth=2, color='blue', alpha=0.7)
            
            if has_eval:
                eval_data = history[['_step', 'eval/loss']].dropna()
                if len(eval_data) > 0:
                    ax.plot(eval_data['_step'], eval_data['eval/loss'], 
                            label='Validation Loss', linewidth=2, color='red', alpha=0.7)
        
        ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig7_training_curves_best_run.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_DIR}/fig7_training_curves_best_run.png")

# Figure 8: Hyperparameter Search Space
print("\n[8/8] Generating hyperparameter exploration plot...")
df_valid = df[df['eval_loss'].notna()]
if len(df_valid) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    hyperparams_viz = [
        ('learning_rate', 'Learning Rate', 'log'),
        ('effective_batch', 'Effective Batch Size', 'linear'),
        ('warmup_steps', 'Warmup Steps', 'linear'),
        ('weight_decay', 'Weight Decay', 'log'),
        ('epochs', 'Num Epochs', 'linear'),
        ('scheduler', 'LR Scheduler', 'categorical')
    ]
    
    for idx, (param, title, scale) in enumerate(hyperparams_viz):
        ax = axes[idx]
        
        if param == 'scheduler':
            counts = df_valid[param].value_counts()
            ax.bar(counts.index, counts.values, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Number of Runs', fontweight='bold')
        else:
            data = df_valid[[param, 'eval_loss']].dropna()
            if len(data) > 0:
                scatter = ax.scatter(data[param], data['eval_loss'], 
                                    c=data['eval_loss'], cmap='RdYlGn_r',
                                    s=150, alpha=0.7, edgecolors='black', linewidth=1)
                ax.set_ylabel('Validation Loss', fontweight='bold')
                
                if scale == 'log':
                    ax.set_xscale('log')
        
        ax.set_xlabel(title, fontweight='bold')
        ax.set_title(f'{title} Exploration', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig8_hyperparameter_exploration.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig8_hyperparameter_exploration.png")

# Diagnostic
with open(f"{OUTPUT_DIR}/diagnostic_metrics.txt", "w", encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("WANDB METRICS DIAGNOSTIC\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total runs: {len(df)}\n\n")
    
    f.write("Run states:\n")
    for state in df['state'].unique():
        count = len(df[df['state'] == state])
        f.write(f"  - {state}: {count}\n")
    
    f.write("\nMetrics available per run:\n")
    for idx, row in df.iterrows():
        f.write(f"\n{idx+1}. {row['name']} (state: {row['state']})\n")
        f.write(f"   eval_loss: {row['eval_loss']}\n")
        f.write(f"   spice_has_source: {row['spice_has_source']}\n")
        f.write(f"   spice_has_ground: {row['spice_has_ground']}\n")
        f.write(f"   spice_has_end: {row['spice_has_end']}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("ALL AVAILABLE KEYS (first run):\n")
    f.write("="*80 + "\n")
    if len(sweep.runs) > 0:
        first_run = list(sweep.runs)[0]
        all_keys = sorted(first_run.summary.keys())
        for key in all_keys:
            f.write(f"  - {key}: {first_run.summary.get(key)}\n")

print(f"  ✓ Saved: {OUTPUT_DIR}/diagnostic_metrics.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print(f"  - fig1_sweep_progress.png")
print(f"  - fig2_parallel_coordinates.png")
print(f"  - fig3_hyperparameter_importance.png")
print(f"  - fig4_spice_metrics.png")
print(f"  - fig5_spice_metrics_avg.png")
print(f"  - fig6_top5_comparison.png")
print(f"  - fig7_training_curves_best_run.png")
print(f"  - fig8_hyperparameter_exploration.png")
print(f"  - table_top5_configs.csv")
print(f"  - diagnostic_metrics.txt")
if len(df_top5) > 0:
    print(f"\nBest run: {df_top5.iloc[0]['name']} (loss: {df_top5.iloc[0]['eval_loss']:.4f})")
