# NYU Greene HPC Setup Guide

Quick guide to deploy and run SLM experiments on NYU Greene.

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
scp -r hpc_datasets/ YOUR_NETID@greene.hpc.nyu.edu:/scratch/YOUR_NETID/SLM_system/
scp -r scripts/ YOUR_NETID@greene.hpc.nyu.edu:/scratch/YOUR_NETID/SLM_system/
```

### Create Conda Environment
```bash
module load anaconda3/2024.02
conda create -n slm_env python=3.10 -y
conda activate slm_env
pip install torch transformers datasets pandas tqdm tensorboard scikit-learn accelerate
```

### Update SLURM Scripts
Edit both `scripts/slurm/train_gpt2_grammar.sh` and `scripts/slurm/train_gpt2_readability.sh`:

Replace `YOUR_NETID` with your actual NetID (3 places in each file)
Replace `YOUR_EMAIL@nyu.edu` with your NYU email

**Quick replace:**
```bash
cd /scratch/YOUR_NETID/SLM_system
sed -i 's/YOUR_NETID/YOUR_ACTUAL_NETID/g' scripts/slurm/train_gpt2_grammar.sh
sed -i 's/YOUR_NETID/YOUR_ACTUAL_NETID/g' scripts/slurm/train_gpt2_readability.sh
sed -i 's/YOUR_EMAIL@nyu.edu/YOUR_ACTUAL_EMAIL@nyu.edu/g' scripts/slurm/train_gpt2_grammar.sh
sed -i 's/YOUR_EMAIL@nyu.edu/YOUR_ACTUAL_EMAIL@nyu.edu/g' scripts/slurm/train_gpt2_readability.sh
```

### Create Directories
```bash
mkdir -p logs models results
```

## 2. Running Experiments

### Grammar Correction (BEA-2019 → JFLEG)
```bash
sbatch scripts/slurm/train_gpt2_grammar.sh
```
**Time**: ~8-12 hours  
**Output**: `models/gpt2_grammar/`, `results/grammar_*.csv`

### Text Simplification (WikiAuto → ASSET)
```bash
sbatch scripts/slurm/train_gpt2_readability.sh
```
**Time**: ~24-36 hours  
**Output**: `models/gpt2_readability/`, `results/readability_*.csv`

## 3. Monitoring Jobs

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

## 4. Collecting Results

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

## 5. Troubleshooting

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

## 6. Expected Results

### Grammar Task
- **Training**: 68,616 examples (BEA-2019)
- **Validation**: 8,768 examples (BEA dev)
- **Test**: 748 examples (JFLEG)

### Readability Task
- **Training**: 373,801 examples (WikiAuto)
- **Validation**: 2,000 examples (ASSET)
- **Test**: 359 examples (ASSET)

## Notes

- Use `/scratch/` for all computations (not `/home/`)
- Models save checkpoints every 500-1000 steps
- Jobs auto-resume from last checkpoint if interrupted
- TensorBoard logs in `models/*/logs/`

