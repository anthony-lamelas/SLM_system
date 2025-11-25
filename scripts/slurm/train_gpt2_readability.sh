#!/bin/bash
#SBATCH --job-name=gpt2_readability
#SBATCH --output=logs/gpt2_readability_%j.out
#SBATCH --error=logs/gpt2_readability_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Mail alerts when the job ends
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu

# Load required modules for NYU Greene
module load anaconda3/2024.02
module load cuda/11.3.1

# Activate your conda environment
source ~/.bashrc
conda activate /scratch/YOUR_NETID/conda_envs/slm_env

# Change to your working directory on scratch
cd /scratch/YOUR_NETID/SLM_system

# Create logs and results directories if they don't exist
mkdir -p logs results

# Train GPT-2 on WikiAuto for text simplification
python scripts/python/gpt-2-readability/train_gpt2_readability.py \
    --train_data /scratch/YOUR_NETID/SLM_system/hpc_datasets/wikiauto_train.csv \
    --model_size gpt2 \
    --output_dir /scratch/YOUR_NETID/SLM_system/models/gpt2_readability \
    --epochs 3 \
    --batch_size 4

# Evaluate on ASSET validation set
python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
    --model_path /scratch/YOUR_NETID/SLM_system/models/gpt2_readability/final_model \
    --test_data /scratch/YOUR_NETID/SLM_system/hpc_datasets/asset_validation.csv \
    --output_path /scratch/YOUR_NETID/SLM_system/results/readability_asset_validation.csv

# Evaluate on ASSET test set
python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
    --model_path /scratch/YOUR_NETID/SLM_system/models/gpt2_readability/final_model \
    --test_data /scratch/YOUR_NETID/SLM_system/hpc_datasets/asset_test.csv \
    --output_path /scratch/YOUR_NETID/SLM_system/results/readability_asset_test.csv
