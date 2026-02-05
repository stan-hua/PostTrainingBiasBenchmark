#!/bin/bash -l
#SBATCH --job-name=ceb_delete                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=2                 # Number of CPU cores per TASK
#SBATCH --mem=4GB
#SBATCH --tmp=1GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=10:00:00


################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
pixi shell -e analysis

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn


################################################################################
#                                 Delete Data                                  #
################################################################################
DSETS=(
    "CEB-Adult"
    "CEB-Jigsaw"
    "CEB-Credit"
    "CEB-Recognition-*"
    "CEB-Selection-*"
)
for DSET in "${DSETS[@]}"; do
    python -m scripts.benchmark delete --dataset_regex $DSET --file_regex "*.json" --inference;
done
