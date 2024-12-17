#!/bin/bash

# Define the configs and job names
declare -a configs=(
    "configs/official/OLMo-1B-50prompts50synth.yaml"
    "configs/official/OLMo-1B-100prompts.yaml"
    "configs/official/OLMo-1B-100synth.yaml"
)

declare -a job_names=(
    "OLMo-1B-50prompts50synth"
    "OLMo-1B-100prompts"
    "OLMo-1B-100synth"
)

# Loop through configs and submit jobs
for i in "${!configs[@]}"; do
    config="${configs[i]}"
    job_name="${job_names[i]}"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name         # Name of the job
#SBATCH --output=/n/home07/bham/mid-OLMo/logs/%j.out          # Output log file (%j is the job ID)
#SBATCH --error=/n/home07/bham/mid-OLMo/logs/%j.err           # Error log file
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=4           # Number of tasks per node
#SBATCH --gpus-per-node=4             # GPUs per node
#SBATCH --mem=256GB                   # Total memory per node
#SBATCH --time=48:00:00               # Time limit for the job
#SBATCH --constraint=h100
#SBATCH --partition=kempner_h100  # Partition name (replace with your cluster's partition)
#SBATCH --cpus-per-task=24         # Number of CPUs per task

# Run the training script
srun torchrun -m --nproc_per_node=4 scripts.train $config --save-overwrite
EOT

    echo "Submitted job for config: $config with job name: $job_name"
done