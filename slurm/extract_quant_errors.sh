#!/bin/bash -l
#SBATCH --job-name=extract_quant_errors                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per TASK
#SBATCH --mem=8GB
#SBATCH --tmp=8GB
#SBATCH -o slurm/logs/audit_datasets-%j.out
#SBATCH --time=10:00:00

# If you want to do it in the terminal,
#--gres=gpu:NVIDIA_L40S:1                      # Request one GPU
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_L40S:1 --cpus-per-task=2 --mem=16G --tmp 8GB
# srun (command)


################################################################################
#                                 Environment                                  #
################################################################################
# # Load CUDA libraries
# module load gcc/12.1.0
# module load cuda/12.4.1

# # Load any necessary modules or activate your virtual environment here
# micromamba activate fairbench

# (Pixi) Load environment
pixi shell -e ct

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn


################################################################################
#                                   Scripts                                    #
################################################################################
python -m scripts.extract_quant_errors
