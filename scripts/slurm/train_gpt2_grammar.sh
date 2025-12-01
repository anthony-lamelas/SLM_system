#!/bin/bash
#SBATCH --job-name=grammar
#SBATCH --output=logs/grammar_%j.out
#SBATCH --error=logs/grammar_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Mail alerts when the job ends
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu

# Usage: sbatch train_gpt2_grammar.sh <model_name> [--use_gradient_checkpointing]
# Example: sbatch train_gpt2_grammar.sh gpt2-medium
# Example: sbatch train_gpt2_grammar.sh EleutherAI/gpt-neo-1.3B --use_gradient_checkpointing

MODEL_NAME=${1:-gpt2}  # Default to gpt2 if not provided
USE_GRADIENT_CHECKPOINTING=${2:-""}  # Optional flag

# Create a safe model identifier for directory names (replace / with _)
MODEL_ID=$(echo "$MODEL_NAME" | sed 's/\//_/g' | sed 's/\./-/g')
OUTPUT_DIR="models/${MODEL_ID}_grammar"

# Set batch size based on model size (adjust if needed)
if [[ "$MODEL_NAME" == *"1.3B"* ]] || [[ "$MODEL_NAME" == *"large"* ]]; then
    BATCH_SIZE=4
else
    BATCH_SIZE=8
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

# Train on BEA dataset for grammatical error correction
python scripts/python/gpt-2-grammar/train_gpt2.py \
    --train_data hpc_datasets/bea_train.csv \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --batch_size "$BATCH_SIZE" \
    $USE_GRADIENT_CHECKPOINTING

# Only evaluate if training succeeded
if [ $? -eq 0 ] && [ -d "$OUTPUT_DIR/final_model" ]; then
    echo "Training completed successfully. Starting evaluation..."
    
    # Evaluate on BEA dev set
    python scripts/python/gpt-2-grammar/evaluate_gpt2.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --test_data hpc_datasets/bea_dev.csv \
        --output_path "results/grammar_${MODEL_ID}_bea_dev.csv"
    
    # Evaluate on JFLEG test set
    python scripts/python/gpt-2-grammar/evaluate_gpt2.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --test_data hpc_datasets/jfleg_test.csv \
        --output_path "results/grammar_${MODEL_ID}_jfleg_test.csv"
else
    echo "Training failed or model not found. Skipping evaluation."
    exit 1
fi
