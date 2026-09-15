# SPICE Circuit Generator - Training Code

This repository contains the complete source code for training a T5-based transformer model to generate SPICE circuit netlists from natural language descriptions.

## 📁 Project Structure

```
CODE/
├── training/              # Model training scripts
├── visualization/         # Analysis and visualization tools
├── data_generator/        # Training data generation
│   ├── chatgpt/          # GPT-4 based data generation
│   └── synthetic/        # Rule-based synthetic data generation
└── requirements.txt       # Python dependencies
```

## 🔧 Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## 📚 Directory Details

### 1. `training/` - Model Training

Contains scripts for training the T5 model with different strategies:

-   **`train.py`**: Basic training script for single model training

    -   Loads dataset from CSV
    -   Configures T5-small model with custom hyperparameters
    -   Trains with WandB logging
    -   Saves final model to `spice_model_final/`

-   **`run_sweep.py`**: Hyperparameter sweep orchestrator

    -   Configures Bayesian optimization sweep
    -   Defines search space for hyperparameters
    -   Launches WandB sweep agents

-   **`train_with_sweep.py`**: Training function for sweep runs
    -   Extends Trainer with custom SPICE validation metrics
    -   Compatible with WandB agent for hyperparameter optimization
    -   Validates generated netlists during training
    -   Saves best models to `best_models/`

**Usage:**

```bash
# Single training run
python training/train.py

# Hyperparameter sweep
python training/run_sweep.py
```

### 2. `visualization/` - Results Analysis

Tools for analyzing training results and generating report figures:

-   **`analyze_best_run.py`**: Detailed analysis of best training run

    -   Downloads training history from WandB
    -   Generates loss curves, learning rate schedule, SPICE metrics evolution
    -   Creates gradient norm plots for stability analysis
    -   Exports statistics for report (Section 5.2)
    -   Output: `visualization/output/`

-   **`analyze_sweep_results.py`**: Comprehensive sweep analysis
    -   Analyzes all runs in a WandB sweep
    -   Generates parallel coordinates plot
    -   Computes hyperparameter importance
    -   Creates SPICE metrics breakdown
    -   Identifies top 5 configurations
    -   Output: `visualization/output/`

**Usage:**

```bash
# Analyze best run
python visualization/analyze_best_run.py

# Analyze entire sweep
python visualization/analyze_sweep_results.py
```

**Output:** All figures are saved to `visualization/output/` in high-resolution PNG format.

### 3. `data_generator/` - Training Data Generation

Two complementary approaches for generating training data:

#### 3.1 `chatgpt/` - GPT-4 Based Generation

Uses OpenAI's GPT-4 API to generate diverse, natural circuit descriptions paired with Netlist SPICE files.

**Files:**

-   `main.py`: Async orchestrator for batch generation
-   `services/chatgpt.py`: GPT-4 agent with detailed prompt engineering

**Features:**

-   Generates realistic natural language descriptions
-   Creates valid Netlist SPICE schematics
-   Asynchronous batch processing
-   Exports to CSV format

**Usage:**

```bash
cd data_generator/chatgpt
# Set OPENAI_API_KEY in .env file
python main.py
```

**Output:** `results.csv` with columns: `[description, asc]`

#### 3.2 `synthetic/` - Rule-Based Synthetic Generation

Programmatically generates SPICE netlists using randomized component values and circuit topologies.

**Files:**

-   `rc.py`: RC filter circuits (low-pass, high-pass, 1-3 stages)
-   `bjt_amp.py`: BJT amplifier circuits (CE, CC configurations)
-   `mos_amp.py`: MOSFET amplifier circuits (CS, CD, CG)
-   `cascaded.py`: Multi-stage cascaded circuits
-   `feedback.py`: Feedback networks (R, C, RC feedback)
-   `export_csv.py`: Main script to generate and export dataset

**Features:**

-   20+ circuit topology variations
-   Randomized component values from realistic ranges
-   Structured natural language descriptions
-   Deterministic with seed control
-   Generates 13,000+ samples by default

**Configuration:**
Edit `export_csv.py` to adjust sample counts:

```python
dataset = []
dataset += rc_gen(5000)        # RC filters
dataset += cascaded_gen(2000)  # Cascaded circuits
dataset += feedback_gen(2000)  # Feedback circuits
dataset += bjt_gen(2000)       # BJT amplifiers
dataset += mos_gen(2000)       # MOS amplifiers
```

**Usage:**

```bash
cd data_generator/synthetic
python export_csv.py
```

**Output:** `results_augmented.csv` with columns: `[input_text, output_text]`

## 🎯 Typical Workflow

1. **Generate Training Data:**

    ```bash
    # Option A: Synthetic generation
    cd data_generator/synthetic
    python export_csv.py

    # Option B: GPT-4 generation
    cd data_generator/chatgpt
    python main.py
    ```

2. **Train Model:**

    ```bash
    # Single training run
    python training/train.py

    # Or hyperparameter sweep
    python training/run_sweep.py
    ```

3. **Analyze Results:**

    ```bash
    # Analyze best run
    python visualization/analyze_best_run.py

    # Analyze all sweep runs
    python visualization/analyze_sweep_results.py
    ```

## 📊 Key Outputs

-   **Trained Models:** `spice_model_final/` or `best_models/[run-name]/`
-   **Training Figures:** `visualization/output/`
-   **Training Data:** `results_augmented.csv`
-   **WandB Logs:** Available on WandB dashboard

## 🔑 Configuration

### WandB Setup

```bash
wandb login
# Enter your API key when prompted
```

### OpenAI API (for ChatGPT data generation)

Create `.env` file in `data_generator/chatgpt/`:

```
OPENAI_API_KEY=your_api_key_here
```
