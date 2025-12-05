#!/bin/bash

# Retrain 4 missing models with aggressive hyperparameters

# 1. gpt2_readability
sbatch --job-name=gpt2_read \
  --output=logs/train_gpt2_read_%j.out \
  --error=logs/train_gpt2_read_%j.err \
  --time=72:00:00 \
  --gres=gpu:2 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --mail-type=END \
  --mail-user=al8372@nyu.edu \
  --wrap="source ~/.bashrc && conda activate /scratch/al8372/conda_envs/slm_env && cd /scratch/al8372/SLM_system && python scripts/python/gpt-2-readability/train_gpt2_readability.py --model_name gpt2 --train_data hpc_datasets/asset_train.csv --val_data hpc_datasets/asset_val.csv --output_dir models/gpt2_readability --num_epochs 10 --batch_size 4 --learning_rate 3e-5 --use_fp16"

# 2. gpt2-medium_readability
sbatch --job-name=gpt2med_read \
  --output=logs/train_gpt2med_read_%j.out \
  --error=logs/train_gpt2med_read_%j.err \
  --time=72:00:00 \
  --gres=gpu:2 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --mail-type=END \
  --mail-user=al8372@nyu.edu \
  --wrap="source ~/.bashrc && conda activate /scratch/al8372/conda_envs/slm_env && cd /scratch/al8372/SLM_system && python scripts/python/gpt-2-readability/train_gpt2_readability.py --model_name gpt2-medium --train_data hpc_datasets/asset_train.csv --val_data hpc_datasets/asset_val.csv --output_dir models/gpt2-medium_readability --num_epochs 10 --batch_size 2 --learning_rate 2e-5 --use_fp16 --gradient_accumulation_steps 2"

# 3. gpt2-large_grammar
sbatch --job-name=gpt2large_gram \
  --output=logs/train_gpt2large_gram_%j.out \
  --error=logs/train_gpt2large_gram_%j.err \
  --time=72:00:00 \
  --gres=gpu:2 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --mail-type=END \
  --mail-user=al8372@nyu.edu \
  --wrap="source ~/.bashrc && conda activate /scratch/al8372/conda_envs/slm_env && cd /scratch/al8372/SLM_system && python scripts/python/gpt-2-grammar/train_gpt2_grammar.py --model_name gpt2-large --train_data hpc_datasets/jfleg_train.csv --val_data hpc_datasets/jfleg_val.csv --output_dir models/gpt2-large_grammar --num_epochs 10 --batch_size 2 --learning_rate 2e-5 --use_fp16 --gradient_accumulation_steps 2 --gradient_checkpointing"

# 4. gpt2-large_readability
sbatch --job-name=gpt2large_read \
  --output=logs/train_gpt2large_read_%j.out \
  --error=logs/train_gpt2large_read_%j.err \
  --time=72:00:00 \
  --gres=gpu:2 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --mail-type=END \
  --mail-user=al8372@nyu.edu \
  --wrap="source ~/.bashrc && conda activate /scratch/al8372/conda_envs/slm_env && cd /scratch/al8372/SLM_system && python scripts/python/gpt-2-readability/train_gpt2_readability.py --model_name gpt2-large --train_data hpc_datasets/asset_train.csv --val_data hpc_datasets/asset_val.csv --output_dir models/gpt2-large_readability --num_epochs 10 --batch_size 2 --learning_rate 2e-5 --use_fp16 --gradient_accumulation_steps 2 --gradient_checkpointing"

echo "Submitted 4 training jobs"

