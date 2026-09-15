import wandb
import sys

PROJECT_NAME = "t5-spice-generator-v4"
NUM_RUNS = 3

SWEEP_CONFIG = {
    "name": "hyperparameter-optimization",
    "method": "bayes",
    "metric": {
        "name": "eval/loss",
        "goal": "minimize"
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "eta": 2
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4
        },
        "per_device_train_batch_size": {
            "values": [64, 128, 256]
        },
        "gradient_accumulation_steps": {
            "values": [1, 2, 4]
        },
        "warmup_steps": {
            "values": [500, 1000, 1500]
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.1
        },
        "num_train_epochs": {
            "values": [10, 15, 20]
        },
        "lr_scheduler_type": {
            "values": ["cosine", "linear", "polynomial"]
        }
    }
}

def main():
    print("="*20)
    print("LAUNCHING HYPERPARAMETER SWEEP")
    print("="*20)
    print(f"\nProject: {PROJECT_NAME}")
    print(f"Method: {SWEEP_CONFIG['method'].upper()}")
    print(f"Metric: {SWEEP_CONFIG['metric']['name']} (goal: {SWEEP_CONFIG['metric']['goal']})")
    
    print("\nHYPERPARAMETERS TO OPTIMIZE:")
    for param, config in SWEEP_CONFIG['parameters'].items():
        if 'values' in config:
            print(f"  - {param}: {config['values']}")
        elif 'distribution' in config:
            min_val = config.get('min', '')
            max_val = config.get('max', '')
            print(f"  - {param}: {config['distribution']} [{min_val} - {max_val}]")
    
    print("\n" + "="*20)
    print("Creating sweep on wandb.ai...")
    print("="*20)
    
    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project=PROJECT_NAME
    )
    
    print(f"\nSweep created successfully!")
    print(f"Sweep ID: {sweep_id}")
    print(f"Dashboard: https://wandb.ai/{wandb.api.default_entity}/{PROJECT_NAME}/sweeps/{sweep_id}")
    
    print(f"\nExecuting {NUM_RUNS} runs")
    
    from train_with_sweep import train
    
    print("\n" + "="*20)
    print("STARTING SWEEP")
    print("="*20 + "\n")
    
    wandb.agent(
        sweep_id,
        function=train,
        count=NUM_RUNS,
        project=PROJECT_NAME
    )
    
    print("\n" + "="*20)
    print("SWEEP COMPLETED!")
    print("="*20)
    print(f"\nView results at:")
    print(f"   https://wandb.ai/{wandb.api.default_entity}/{PROJECT_NAME}/sweeps/{sweep_id}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
