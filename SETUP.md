# NYU Greene HPC Setup Guide

Quick guide to deploy and run SLM experiments on NYU Greene.


#### keygen reset if errors
```bash
# Remove the old host key (as suggested in the error message)
ssh-keygen -f '/home/dell/.ssh/known_hosts' -R 'greene.hpc.nyu.edu'
```

## 1. Initial Setup (One-time)

### Login to Greene
```bash
ssh YOUR_NETID@greene.hpc.nyu.edu
```

### Create Project Directory
```bash
mkdir -p /scratch/YOUR_NETID/SLM_system
cd /scratch/YOUR_NETID/SLM_system
```

### Transfer Files from Local
```bash
# From your local machine
# Transfer data directory (contains all CSV datasets: bea_train.csv, bea_dev.csv, jfleg_test.csv, etc.)
scp -r data/ YOUR_NETID@greene.hpc.nyu.edu:/scratch/YOUR_NETID/SLM_system/
# Transfer scripts directory (contains Python training scripts and SLURM job scripts)
scp -r scripts/ YOUR_NETID@greene.hpc.nyu.edu:/scratch/YOUR_NETID/SLM_system/
```

**Files to transfer:**
- `data/` directory: Contains all 6 CSV dataset files (bea_train.csv, bea_dev.csv, jfleg_test.csv, wikiauto_train.csv, asset_validation.csv, asset_test.csv)
- `scripts/` directory: Contains Python training/evaluation scripts and SLURM job submission scripts

### Rename Data Directory on HPC
The scripts expect `hpc_datasets/` but data is in `data/`. Rename it:
```bash
# On Greene HPC (after transferring files)
cd /scratch/YOUR_NETID/SLM_system
mv data hpc_datasets
```

### Create Conda Environment
```bash
module load anaconda3/2024.02
conda create -n slm_env python=3.10 -y
conda init
# restart shell
conda activate slm_env
pip install torch transformers datasets pandas tqdm tensorboard scikit-learn accelerate

# Note: Save the conda environment path for later
# It will be something like: /vast/YOUR_NETID/conda_envs/slm_env
# Update this path in the SLURM scripts (line 22) if different
```

### Get Your Project ID (Required for Tandon HPC)

NYU Tandon HPC requires a project ID to submit jobs. The format is `pr_XXX`.

**How to get your Project ID:**
1. **If you're a researcher/student**: Request your Project ID from your PI (Principal Investigator)
2. **If you're a PI**: Register a project at https://projects.rit.nyu.edu/
3. **For help**: Email soehelpdesk@nyu.edu

**Check if you already have a project ID:**
```bash
# On Greene HPC
sacctmgr show user $USER
```

### Update SLURM Scripts
Edit both `scripts/slurm/train_gpt2_grammar.sh` and `scripts/slurm/train_gpt2_readability.sh`:

Replace `YOUR_NETID` with your actual NetID
Replace `YOUR_EMAIL@nyu.edu` with your NYU email
Replace `pr_XXX` with your actual project ID (format: pr_XXX)

**Quick replace with your values:**
```bash
# On Greene HPC, after transferring files
# Set your values (replace with your actual NetID, email, and project ID)
export NETID="YOUR_NETID"  # Replace with your NetID
export EMAIL="your.email@nyu.edu"  # Replace with your NYU email
export PROJECT_ID="pr_XXX"  # Replace with your project ID (e.g., pr_123)

cd /scratch/$NETID/SLM_system

# Replace YOUR_NETID, YOUR_EMAIL, and pr_XXX in both SLURM scripts
sed -i "s/YOUR_NETID/$NETID/g" scripts/slurm/train_gpt2_grammar.sh
sed -i "s/YOUR_NETID/$NETID/g" scripts/slurm/train_gpt2_readability.sh
sed -i "s/YOUR_EMAIL@nyu.edu/$EMAIL/g" scripts/slurm/train_gpt2_grammar.sh
sed -i "s/YOUR_EMAIL@nyu.edu/$EMAIL/g" scripts/slurm/train_gpt2_readability.sh
sed -i "s/pr_XXX/$PROJECT_ID/g" scripts/slurm/train_gpt2_grammar.sh
sed -i "s/pr_XXX/$PROJECT_ID/g" scripts/slurm/train_gpt2_readability.sh

# Verify the changes (should return nothing if successful)
grep -n "YOUR_NETID\|YOUR_EMAIL\|pr_XXX" scripts/slurm/*.sh
```

**Or one-liner (replace with your actual values):**
```bash
cd /scratch/YOUR_NETID/SLM_system && sed -i 's/YOUR_NETID/YOUR_ACTUAL_NETID/g; s/YOUR_EMAIL@nyu.edu/your.email@nyu.edu/g; s/pr_XXX/pr_YOUR_PROJECT/g' scripts/slurm/*.sh
```

### Create Directories
```bash
mkdir -p logs models results
```

### Find Your Conda Environment Path
After creating the conda environment, find its full path:
```bash
conda env list | grep slm_env
# or
conda info --envs
```
The path will be something like: `/vast/YOUR_NETID/conda_envs/slm_env`

**Update line 22 in both SLURM scripts** with your actual conda environment path.

## 2. Complete Workflow & Output Locations

### Workflow Overview

Each SLURM script does **3 steps automatically**:

1. **Train** the model on training data
2. **Evaluate** on validation set
3. **Evaluate** on test set

**You don't need separate evaluation scripts** - evaluation is already included!

### Grammar Correction Workflow (`train_gpt2_grammar.sh`)

**Step 1: Training**
- Trains GPT-2 on BEA-2019 training set (68,616 examples)
- Saves model checkpoints and final model

**Step 2: Validation Evaluation**
- Evaluates on BEA dev set (8,768 examples)
- Generates predictions and metrics

**Step 3: Test Evaluation**
- Evaluates on JFLEG test set (748 examples)
- Generates predictions and metrics

### Text Simplification Workflow (`train_gpt2_readability.sh`)

**Step 1: Training**
- Trains GPT-2 on WikiAuto training set (373,801 examples)
- Saves model checkpoints and final model

**Step 2: Validation Evaluation**
- Evaluates on ASSET validation set (2,000 examples)
- Generates predictions and metrics

**Step 3: Test Evaluation**
- Evaluates on ASSET test set (359 examples)
- Generates predictions and metrics

### Output File Locations

All outputs are saved in `/scratch/YOUR_NETID/SLM_system/`:

#### Training Outputs

**Grammar Model:**
- `models/gpt2_grammar/checkpoint-{step}/` - Training checkpoints (every 500 steps)
- `models/gpt2_grammar/final_model/` - Final trained model (used for evaluation)
- `models/gpt2_grammar/logs/` - TensorBoard training logs

**Readability Model:**
- `models/gpt2_readability/checkpoint-{step}/` - Training checkpoints (every 1000 steps)
- `models/gpt2_readability/final_model/` - Final trained model (used for evaluation)
- `models/gpt2_readability/logs/` - TensorBoard training logs

#### Evaluation Outputs

**Grammar Task Results:**
- `results/grammar_bea_dev.csv` - Predictions on BEA dev set
  - Columns: `input_text`, `target_text`, `prediction`
- `results/grammar_bea_dev_metrics.json` - Metrics for BEA dev
  - Contains: exact match accuracy, total examples
- `results/grammar_jfleg_test.csv` - Predictions on JFLEG test set
- `results/grammar_jfleg_test_metrics.json` - Metrics for JFLEG test

**Readability Task Results:**
- `results/readability_asset_validation.csv` - Predictions on ASSET validation
  - Columns: `input_text`, `target_text`, `prediction`
- `results/readability_asset_validation_metrics.json` - Metrics for ASSET validation
  - Contains: exact match accuracy, length statistics, compression rates
- `results/readability_asset_test.csv` - Predictions on ASSET test set
- `results/readability_asset_test_metrics.json` - Metrics for ASSET test

#### SLURM Job Logs

- `logs/gpt2_grammar_{JOBID}.out` - Standard output from grammar job
- `logs/gpt2_grammar_{JOBID}.err` - Error output from grammar job
- `logs/gpt2_readability_{JOBID}.out` - Standard output from readability job
- `logs/gpt2_readability_{JOBID}.err` - Error output from readability job

## 3. Running Experiments

### Grammar Correction (BEA-2019 → JFLEG)
```bash
sbatch scripts/slurm/train_gpt2_grammar.sh
```
**Time**: ~8-12 hours  
**Output**: 
- `models/gpt2_grammar/` - Trained model
- `models/gpt2_grammar/training_metadata.json` - Training time and GPU info
- `results/grammar_*.csv` - Evaluation predictions

### Text Simplification (WikiAuto → ASSET)
```bash
sbatch scripts/slurm/train_gpt2_readability.sh
```
**Time**: ~24-36 hours  
**Output**: 
- `models/gpt2_readability/` - Trained model
- `models/gpt2_readability/training_metadata.json` - Training time and GPU info
- `results/readability_*.csv` - Evaluation predictions

## 4. Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View logs (real-time)
```bash
tail -f logs/gpt2_grammar_JOBID.out
tail -f logs/gpt2_readability_JOBID.out
```

### Cancel a job
```bash
scancel JOBID
```

## 5. Collecting Results

### View metrics
```bash
cat results/grammar_bea_dev_metrics.json
cat results/grammar_jfleg_test_metrics.json
cat results/readability_asset_validation_metrics.json
cat results/readability_asset_test_metrics.json
```

### Download results to local machine
```bash
# From your local machine
scp -r YOUR_NETID@greene.hpc.nyu.edu:/scratch/YOUR_NETID/SLM_system/results/ ./
```

## 6. Troubleshooting

### Out of Memory
Reduce batch size in the SLURM script:
```bash
--batch_size 2
```

### Job Timeout
Increase time limit:
```bash
#SBATCH --time=48:00:00
```

### Module Not Found
Check available modules:
```bash
module avail anaconda
module avail cuda
```

### Check Partition Availability
```bash
sinfo -p a100_1,a100_2,v100
```

## 7. Expected Results

### Grammar Task
- **Training**: 68,616 examples (BEA-2019)
- **Validation**: 8,768 examples (BEA dev)
- **Test**: 748 examples (JFLEG)

### Readability Task
- **Training**: 373,801 examples (WikiAuto)
- **Validation**: 2,000 examples (ASSET)
- **Test**: 359 examples (ASSET)

## 8. Training Time Tracking

Training scripts automatically track:
- **Training duration** (hours, minutes, seconds)
- **GPU type and memory usage**
- **Total training steps**
- **Final training loss**

**Metadata saved to**: `models/*/training_metadata.json`

After training completes, view the metadata:
```bash
cat models/gpt2_grammar/training_metadata.json
cat models/gpt2_readability/training_metadata.json
```

## Notes

- Use `/scratch/` for all computations (not `/home/`)
- Models save checkpoints every 500-1000 steps
- Jobs auto-resume from last checkpoint if interrupted
- TensorBoard logs in `models/*/logs/`
- Training metadata (time, GPU info) saved automatically to `models/*/training_metadata.json`

