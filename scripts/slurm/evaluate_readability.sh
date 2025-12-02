#!/bin/bash
#SBATCH --job-name=eval_readability
#SBATCH --output=logs/eval_readability_%j.out
#SBATCH --error=logs/eval_readability_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Mail alerts when the job ends
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu

# Usage: sbatch evaluate_readability.sh <model_name> <test_data> <output_path>
# Example: sbatch evaluate_readability.sh gpt2-medium hpc_datasets/asset_test.csv results/readability_gpt2-medium_asset_test.csv

MODEL_NAME=${1}
TEST_DATA=${2}
OUTPUT_PATH=${3}

# Create a safe model identifier for directory names (replace / with _)
MODEL_ID=$(echo "$MODEL_NAME" | sed 's/\//_/g' | sed 's/\./-/g')
MODEL_PATH="models/${MODEL_ID}_readability/final_model"

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

echo "Evaluating model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Output path: $OUTPUT_PATH"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Run evaluation
python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_DATA" \
    --output_path "$OUTPUT_PATH" \
    --batch_size 8

echo "Evaluation completed!"

