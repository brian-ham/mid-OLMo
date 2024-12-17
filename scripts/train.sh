#!/bin/bash
#SBATCH --job-name=mid-OLMo        # Name of the job
#SBATCH --output=logs/%j.out          # Output log file (%j is the job ID)
#SBATCH --error=logs_%j.err           # Error log file
#SBATCH --account kempner_sham_lab
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=4           # Number of tasks per node
#SBATCH --gpus-per-node=4             # GPUs per node
#SBATCH --mem=256GB                   # Total memory per node
#SBATCH --time=48:00:00               # Time limit for the job
#SBATCH --constraint=h100
#SBATCH --partition=kempner_h100  # Partition name (replace with your cluster's partition)
#SBATCH --cpus-per-task=16         # Number of CPUs per task

# Run the training script
srun torchrun -m --nproc_per_node=4 scripts.train configs/official/OLMo-1B-synth-1B.yaml --save-overwrite