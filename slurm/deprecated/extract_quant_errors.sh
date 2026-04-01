#!/bin/bash -l
#SBATCH --job-name=extract_qerrors                    # Job name
#SBATCH -o slurm/logs/extract_errors-%j.out
#SBATCH --gres=gpu:1                      # Request one GPU
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per TASK
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64GB
# --tmp=8GB

# If you want to do it in the terminal,
# salloc --job-name=ceb --nodes=1 --gres=gpu:1 --cpus-per-task=2 --mem=16G --tmp 8GB
# srun (command)


################################################################################
#                                 Environment                                  #
################################################################################
# # Load any necessary modules or activate your virtual environment here
pixi shell -e analysis

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# List of models
ORIG_MODELS=(
    meta-llama/Llama-3.1-8B-Instruct
    meta-llama/Llama-3.2-1B-Instruct
    meta-llama/Llama-3.2-3B-Instruct
    mistralai/Ministral-8B-Instruct-2410
    Qwen/Qwen2-7B-Instruct
    Qwen/Qwen2.5-0.5B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-3B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct
)


################################################################################
#                                   Scripts                                    #
################################################################################
# Run in pixi environment
for model in "${ORIG_MODELS[@]}"; do
    python -m scripts.extract_quant_errors parallel_analyze \
        --orig_models $model
done
