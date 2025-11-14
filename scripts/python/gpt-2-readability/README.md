# GPT-2 Text Simplification (Readability)

Scripts for training and evaluating GPT-2 on text simplification using WikiAuto for training and ASSET for evaluation.

## Task

**Text Simplification**: Converting complex, formal text into simpler, more accessible language while preserving meaning.

**Training Data**: WikiAuto (373,801 examples)  
**Evaluation Data**: ASSET validation (2,000 examples) and test (359 examples)

## Setup

1. Install dependencies:
```bash
pip install -r scripts/python/gpt-2-readability/requirements.txt
```

2. Ensure datasets are in `hpc_datasets/`:
   - `wikiauto_train.csv`
   - `asset_validation.csv`
   - `asset_test.csv`

## Usage

### 1. Train GPT-2 on WikiAuto

Basic training (full dataset):
```bash
python scripts/python/gpt-2-readability/train_gpt2_readability.py \
    --train_data hpc_datasets/wikiauto_train.csv \
    --model_size gpt2 \
    --output_dir ./models/gpt2_readability \
    --epochs 3
```

Train with limited samples (for testing):
```bash
python scripts/python/gpt-2-readability/train_gpt2_readability.py \
    --train_data hpc_datasets/wikiauto_train.csv \
    --model_size gpt2 \
    --output_dir ./models/gpt2_readability_test \
    --epochs 1 \
    --max_train_samples 10000
```

### 2. Evaluate on ASSET Validation Set

```bash
python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
    --model_path ./models/gpt2_readability/final_model \
    --test_data hpc_datasets/asset_validation.csv \
    --output_path results_asset_validation.csv
```

### 3. Evaluate on ASSET Test Set

```bash
python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
    --model_path ./models/gpt2_readability/final_model \
    --test_data hpc_datasets/asset_test.csv \
    --output_path results_asset_test.csv
```

## Training Arguments

```bash
python scripts/python/gpt-2-readability/train_gpt2_readability.py \
    --train_data <path>              # Path to WikiAuto training CSV
    --model_size gpt2                # Model: gpt2, gpt2-medium, gpt2-large, gpt2-xl
    --output_dir <path>              # Where to save model
    --epochs 3                       # Number of training epochs
    --batch_size 4                   # Per-device batch size
    --learning_rate 5e-5             # Learning rate
    --max_length 512                 # Max sequence length
    --save_steps 1000                # Save checkpoint every N steps
    --max_train_samples <N>          # Limit training samples (optional)
```

## Evaluation Arguments

```bash
python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
    --model_path <path>              # Path to trained model
    --test_data <path>               # Path to ASSET CSV file
    --output_path <path>             # Where to save predictions
    --max_length 512                 # Max sequence length
```

## HPC SLURM Script

Create `train_readability.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=gpt2_readability
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source venv/bin/activate

# Train on WikiAuto
python scripts/python/gpt-2-readability/train_gpt2_readability.py \
    --train_data hpc_datasets/wikiauto_train.csv \
    --model_size gpt2 \
    --output_dir ./models/gpt2_readability \
    --epochs 3 \
    --batch_size 4

# Evaluate on ASSET validation
python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
    --model_path ./models/gpt2_readability/final_model \
    --test_data hpc_datasets/asset_validation.csv \
    --output_path results_asset_validation.csv

# Evaluate on ASSET test
python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
    --model_path ./models/gpt2_readability/final_model \
    --test_data hpc_datasets/asset_test.csv \
    --output_path results_asset_test.csv
```

Submit:
```bash
sbatch train_readability.sh
```

## Output Files

### Training
- Model checkpoints: `./models/gpt2_readability/checkpoint-{step}/`
- Final model: `./models/gpt2_readability/final_model/`
- Training logs: `./models/gpt2_readability/logs/`

### Evaluation
- Predictions CSV: Contains input, target, and model prediction for each example
- Metrics JSON: Contains:
  - Exact match accuracy
  - Average input/prediction/target lengths
  - Compression rates

## Evaluation Metrics

The evaluation script calculates:

1. **Exact Match Accuracy**: % of predictions that exactly match the target
2. **Length Statistics**: Average word counts for input, prediction, and target
3. **Compression Rate**: % reduction in text length from input to simplified output

Example output:
```
EVALUATION METRICS
==================================================
Total examples: 2000
Exact matches: 45
Exact match accuracy: 2.25%

Length Statistics (in words):
  Average input length: 23.4
  Average prediction length: 15.2
  Average target length: 14.8

Compression Rates:
  Prediction compression: 35.0%
  Target compression: 36.8%
==================================================
```

## Expected Training Time

On a single GPU (WikiAuto has 373,801 examples):

- **gpt2** (124M): ~24-36 hours for 3 epochs
- **gpt2-medium** (355M): ~48-72 hours for 3 epochs
- **gpt2-large** (774M): ~72-96 hours for 3 epochs
- **gpt2-xl** (1.5B): ~96+ hours for 3 epochs

## Memory Requirements

- **gpt2**: ~12GB GPU memory (batch_size=4)
- **gpt2-medium**: ~20GB GPU memory (batch_size=4)
- **gpt2-large**: ~32GB GPU memory (batch_size=2-4)
- **gpt2-xl**: ~40GB+ GPU memory (batch_size=2)

Note: WikiAuto is 5x larger than BEA, so expect longer training times.

## Tips

1. **Start with subset**: Use `--max_train_samples 50000` to test on a smaller subset first

2. **Monitor with TensorBoard**:
   ```bash
   tensorboard --logdir ./models/gpt2_readability/logs
   ```

3. **Reduce memory usage**: Lower batch size if needed
   ```bash
   --batch_size 2 --gradient_accumulation_steps 8
   ```

4. **Resume interrupted training**: Trainer automatically resumes from last checkpoint

## Differences from Grammar Correction

- **Task prompt**: "Simplify this text:" vs "Correct this text:"
- **Dataset**: WikiAuto (complex→simple) vs BEA (errors→corrected)
- **Evaluation**: ASSET test sets vs BEA dev/JFLEG
- **Metrics**: Compression rate matters for simplification
- **Training time**: ~5x longer (373k vs 68k examples)
