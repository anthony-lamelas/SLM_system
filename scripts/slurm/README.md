# SLURM Job Scripts

SLURM batch scripts for running GPT-2 training and evaluation on NYU Greene HPC.

## ⚠️ Before Running

**Update these placeholders in both scripts:**
- Replace `YOUR_NETID` with your NYU NetID (3 places per script)
- Replace `YOUR_EMAIL@nyu.edu` with your email

## Available Scripts

### 1. `train_gpt2_grammar.sh`
Train GPT-2 on BEA-2019 for grammatical error correction.

**Submit:**
```bash
sbatch scripts/slurm/train_gpt2_grammar.sh
```

**What it does:**
- Trains GPT-2 on BEA training set (68,616 examples)
- Evaluates on BEA dev set (8,768 examples)
- Evaluates on JFLEG test set (748 examples)
- Time: ~8-12 hours | Memory: 32GB | GPU: 1x (A100/V100)

### 2. `train_gpt2_readability.sh`
Train GPT-2 on WikiAuto for text simplification.

**Submit:**
```bash
sbatch scripts/slurm/train_gpt2_readability.sh
```

**What it does:**
- Trains GPT-2 on WikiAuto training set (373,801 examples)
- Evaluates on ASSET validation set (2,000 examples)
- Evaluates on ASSET test set (359 examples)
- Time: ~24-36 hours | Memory: 64GB | GPU: 1x (A100/V100)

## Customization

### Adjust Resource Requirements

Edit the `#SBATCH` directives in the scripts:

```bash
#SBATCH --time=24:00:00        # Wall time limit
#SBATCH --mem=32G              # Memory
#SBATCH --gres=gpu:1           # Number of GPUs
#SBATCH --cpus-per-task=8      # CPU cores
```

### Change Model Size

Edit the `--model_size` argument:

```bash
# Use larger model
--model_size gpt2-medium

# Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
```

Larger models require more memory and time:
- `gpt2`: 8GB GPU, ~8-12 hours (grammar)
- `gpt2-medium`: 16GB GPU, ~16-24 hours (grammar)
- `gpt2-large`: 24GB GPU, ~24-36 hours (grammar)
- `gpt2-xl`: 32GB GPU, ~48+ hours (grammar)

### Adjust Training Parameters

```bash
--epochs 3              # Number of epochs
--batch_size 8          # Batch size per device
--learning_rate 5e-5    # Learning rate
--max_length 512        # Max sequence length
```

### Module Loading

NYU Greene specific modules (already configured):

```bash
module load anaconda3/2024.02
module load cuda/11.3.1
```

Check available modules on Greene:
```bash
module avail anaconda
module avail cuda
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View job output
```bash
# Real-time monitoring
tail -f slurm-<jobid>.out

# View completed job
cat slurm-<jobid>.out
```

### Cancel job
```bash
scancel <jobid>
```

## Output Files

Both scripts create files in `/scratch/YOUR_NETID/SLM_system/`:

**Training outputs:**
- `models/gpt2_grammar/` or `models/gpt2_readability/`
  - `checkpoint-*/` - Training checkpoints
  - `final_model/` - Final trained model
  - `logs/` - TensorBoard logs

**Evaluation outputs:**
- `results/grammar_*.csv` - Grammar predictions
- `results/readability_*.csv` - Readability predictions
- `results/*_metrics.json` - Evaluation metrics

**Job logs:**
- `logs/gpt2_grammar_JOBID.out` - Standard output
- `logs/gpt2_grammar_JOBID.err` - Error output

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
--batch_size 2
```

### Job Time Limit
Increase time or reduce epochs:
```bash
#SBATCH --time=72:00:00
```

### Wrong Partition
Check available partitions:
```bash
sinfo
```

Update script:
```bash
#SBATCH --partition=<your-gpu-partition>
```

## Example Workflow

1. **Submit grammar training:**
   ```bash
   sbatch scripts/slurm/train_gpt2_grammar.sh
   ```

2. **Monitor progress:**
   ```bash
   squeue -u $USER
   tail -f slurm-<jobid>.out
   ```

3. **Check results when complete:**
   ```bash
   cat results_bea_dev_metrics.json
   ```

4. **Submit readability training:**
   ```bash
   sbatch scripts/slurm/train_gpt2_readability.sh
   ```
