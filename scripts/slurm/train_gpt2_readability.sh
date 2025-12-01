#!/bin/bash
#SBATCH --job-name=readability
#SBATCH --output=logs/readability_%j.out
#SBATCH --error=logs/readability_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Mail alerts when the job ends
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu

# Usage: sbatch train_gpt2_readability.sh <model_name> [--use_gradient_checkpointing]
# Example: sbatch train_gpt2_readability.sh gpt2-medium
# Example: sbatch train_gpt2_readability.sh EleutherAI/gpt-neo-1.3B --use_gradient_checkpointing

MODEL_NAME=${1:-gpt2}  # Default to gpt2 if not provided
USE_GRADIENT_CHECKPOINTING=${2:-""}  # Optional flag

# Create a safe model identifier for directory names (replace / with _)
MODEL_ID=$(echo "$MODEL_NAME" | sed 's/\//_/g' | sed 's/\./-/g')
OUTPUT_DIR="models/${MODEL_ID}_readability"

# Set batch size based on model size (adjust if needed)
if [[ "$MODEL_NAME" == *"1.3B"* ]] || [[ "$MODEL_NAME" == *"2.7B"* ]] || [[ "$MODEL_NAME" == *"large"* ]]; then
    BATCH_SIZE=8
elif [[ "$MODEL_NAME" == *"medium"* ]]; then
    BATCH_SIZE=12
else
    BATCH_SIZE=12
fi

# Load required modules for NYU Greene
module load anaconda3/2024.02
module load cuda/11.3.1

# Activate your conda environment
source ~/.bashrc
conda activate /scratch/YOUR_NETID/conda_envs/slm_env

# Change to your working directory on scratch
cd /scratch/YOUR_NETID/SLM_system

# Create logs and results directories if they don't exist
mkdir -p logs results "$OUTPUT_DIR"

echo "Training model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"

# Train on WikiAuto for text simplification
python scripts/python/gpt-2-readability/train_gpt2_readability.py \
    --train_data hpc_datasets/wikiauto_train.csv \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --batch_size "$BATCH_SIZE" \
    $USE_GRADIENT_CHECKPOINTING

# Only evaluate if training succeeded
if [ $? -eq 0 ] && [ -d "$OUTPUT_DIR/final_model" ]; then
    echo "Training completed successfully. Starting evaluation..."
    
    # Evaluate on ASSET validation set
    python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --test_data hpc_datasets/asset_validation.csv \
        --output_path "results/readability_${MODEL_ID}_asset_validation.csv"
    
    # Evaluate on ASSET test set
    python scripts/python/gpt-2-readability/evaluate_gpt2_readability.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --test_data hpc_datasets/asset_test.csv \
        --output_path "results/readability_${MODEL_ID}_asset_test.csv"
else
    echo "Training failed or model not found. Skipping evaluation."
    exit 1
fi
