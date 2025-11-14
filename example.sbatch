#!/bin/bash
#SBATCH --job-name=layerwise_gpt2_finetune # Job name
#SBATCH --output=logs/fine_tune_%j.out # Standard output log (%j expands to job ID)
#SBATCH --error=logs/fine_tune_%j.err # Standard error log
#SBATCH --time=12:00:00 # Wall-clock time limit (HH:MM:SS)
#SBATCH --partition=a100_1,a100_2,v100 # Partition name; adjust if needed for your HPC
#SBATCH --gres=gpu:1 # Request 1 GPU (adjust if you need more)
#SBATCH --ntasks=1 # Single-task job
#SBATCH --mem=16G # Total memory requested

# Mail alerts when the job ends
#SBATCH --mail-type=END
#SBATCH --mail-user=al8372@nyu.edu

# Optional: set the number of OpenMP threads (tweak based on your workload and nodes)
export OMP_NUM_THREADS=4

# Load required modules (modify module names/versions as available on NYU HPC)
module load anaconda3/2024.02
module load cuda/11.3.1

# Activate your conda environment that has PyTorch, transformers, and other dependencies installed.
# Replace "ml_env" with your environment name.
source ~/.bashrc
conda activate /vast/hem9984/conda_envs/ml_env


# (Optional) Change to the directory where your training script resides.
cd /scratch/hem9984/finetune_llm/data

# Logs
# mkdir -p logs

# Run the training script.
# Adjust the command-line arguments (--tar_file, --output_dir, etc.) as needed.
python train.py --tar_file /scratch/hem9984/finetune_llm/data/data.tar.gz \
--output_dir /scratch/hem9984/finetune_llm/data/output \
--model_name gpt2 \
--aux_layer_index 6 \
--aux_loss_weight 1.0 \
--max_length 512 \
--stride 256 \
--batch_size 4 \
--num_epochs 3 \
--learning_rate 5e-5