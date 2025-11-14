# GPT-2 Training and Evaluation Scripts

Scripts for training and evaluating GPT-2 on grammatical error correction and text simplification tasks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your datasets are in the `hpc_datasets/` directory

## Usage

### 1. Train GPT-2 on BEA Training Set

Basic training with default settings (gpt2-base, 3 epochs):
```bash
python scripts/train_gpt2.py
```

Train with custom settings:
```bash
python scripts/train_gpt2.py \
    --train_data hpc_datasets/bea_train.csv \
    --model_size gpt2 \
    --output_dir ./models/gpt2_bea \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5
```

Model size options:
- `gpt2` (124M params) - fastest, least memory
- `gpt2-medium` (355M params)
- `gpt2-large` (774M params)
- `gpt2-xl` (1.5B params) - best quality, most resources

### 2. Evaluate on BEA Validation Set

```bash
python scripts/evaluate_gpt2.py \
    --model_path ./models/gpt2_bea/final_model \
    --test_data hpc_datasets/bea_dev.csv \
    --output_path results_bea_dev.csv
```

### 3. Evaluate on All Test Sets

Run on BEA dev, JFLEG, WikiAuto, and ASSET:
```bash
python scripts/run_all_tests.py \
    --model_path ./models/gpt2_bea/final_model \
    --output_dir ./results
```

## HPC Usage

### SLURM Job Script Example

Create `train_gpt2.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=gpt2_train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source venv/bin/activate

# Run training
python scripts/train_gpt2.py \
    --train_data hpc_datasets/bea_train.csv \
    --model_size gpt2 \
    --output_dir ./models/gpt2_bea \
    --epochs 3 \
    --batch_size 8

# Run evaluation
python scripts/run_all_tests.py \
    --model_path ./models/gpt2_bea/final_model \
    --output_dir ./results
```

Submit:
```bash
sbatch train_gpt2.sh
```

## Output

### Training
- Checkpoints: `./models/gpt2_bea/checkpoint-{step}/`
- Final model: `./models/gpt2_bea/final_model/`
- Logs: `./models/gpt2_bea/logs/`

### Evaluation
- Predictions CSV: Contains input, target, and prediction for each example
- Metrics JSON: Contains accuracy and other metrics
- Summary JSON: Combined metrics for all datasets

## Expected Training Time

On a single GPU:
- **gpt2** (124M): ~8-12 hours on BEA train (68k examples)
- **gpt2-medium** (355M): ~16-24 hours
- **gpt2-large** (774M): ~24-36 hours
- **gpt2-xl** (1.5B): ~48+ hours

## Memory Requirements

- **gpt2**: ~8GB GPU memory
- **gpt2-medium**: ~16GB GPU memory
- **gpt2-large**: ~24GB GPU memory
- **gpt2-xl**: ~32GB GPU memory

Adjust `--batch_size` if you run out of memory.

## Tips

1. **Monitor training**: Use TensorBoard
   ```bash
   tensorboard --logdir ./models/gpt2_bea/logs
   ```

2. **Resume training**: Training automatically resumes from checkpoints if interrupted

3. **Reduce memory**: Lower batch size or use gradient accumulation:
   ```bash
   --batch_size 2 --gradient_accumulation_steps 8
   ```

4. **Test quickly**: Use a smaller subset first to verify everything works:
   ```bash
   head -1000 hpc_datasets/bea_train.csv > hpc_datasets/bea_train_small.csv
   ```
