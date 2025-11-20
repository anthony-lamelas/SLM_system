#!/bin/bash
#SBATCH --job-name=gpt2_grammar
#SBATCH --output=logs/gpt2_grammar_%j.out
#SBATCH --error=logs/gpt2_grammar_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Mail alerts when the job ends
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu

# Load required modules for NYU Greene
module load anaconda3/2024.02
module load cuda/11.3.1

# Activate your conda environment
source ~/.bashrc
conda activate slm_env

# Change to your working directory on scratch
cd /scratch/YOUR_NETID/SLM_system

# Create logs and results directories if they don't exist
mkdir -p logs results

# Train GPT-2 on BEA dataset for grammatical error correction
python scripts/python/gpt-2-grammar/train_gpt2.py \
    --train_data /scratch/YOUR_NETID/SLM_system/hpc_datasets/bea_train.csv \
    --model_size gpt2 \
    --output_dir /scratch/YOUR_NETID/SLM_system/models/gpt2_grammar \
    --epochs 3 \
    --batch_size 8

# Evaluate on BEA dev set
python scripts/python/gpt-2-grammar/evaluate_gpt2.py \
    --model_path /scratch/YOUR_NETID/SLM_system/models/gpt2_grammar/final_model \
    --test_data /scratch/YOUR_NETID/SLM_system/hpc_datasets/bea_dev.csv \
    --output_path /scratch/YOUR_NETID/SLM_system/results/grammar_bea_dev.csv

# Evaluate on JFLEG test set
python scripts/python/gpt-2-grammar/evaluate_gpt2.py \
    --model_path /scratch/YOUR_NETID/SLM_system/models/gpt2_grammar/final_model \
    --test_data /scratch/YOUR_NETID/SLM_system/hpc_datasets/jfleg_test.csv \
    --output_path /scratch/YOUR_NETID/SLM_system/results/grammar_jfleg_test.csv
