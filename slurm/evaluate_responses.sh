#!/bin/bash -l
#SBATCH --job-name=eval_bias                    # Job name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per TASK
#SBATCH --mem=24GB
#SBATCH -o slurm/logs/slurm-eval_bias-%j.out
#SBATCH --time=24:00:00
# --begin=now+10minutes

# If you want to do it in the terminal,
# salloc --job-name=ceb --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --tmp 8GB
# srun (command)


################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
pixi shell -e vllm

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn


################################################################################
#                                Set Constants                                 #
################################################################################
port=$(shuf -i 6000-9000 -n 1)
echo $port

ALL_MODELS=(
    # # # 2.0. LLaMA 3.2 1B model
    llama3.2-1b
    llama3.2-1b-instruct
    llama3.2-1b-instruct-lc-rtn-w4a16
    llama3.2-1b-instruct-lc-smooth-rtn-w4a16
    llama3.2-1b-instruct-lc-rtn-w8a16
    llama3.2-1b-instruct-lc-gptq-w4a16
    llama3.2-1b-instruct-awq-w4a16

    # # # 2.1. LLaMA 3.2 3B model
    llama3.2-3b
    llama3.2-3b-instruct
    llama3.2-3b-instruct-lc-rtn-w4a16
    llama3.2-3b-instruct-lc-smooth-rtn-w4a16
    llama3.2-3b-instruct-lc-rtn-w8a16
    llama3.2-3b-instruct-lc-gptq-w4a16
    llama3.2-3b-instruct-awq-w4a16

    # # # 2.2. LLaMA 3.1 8B model
    llama3.1-8b
    llama3.1-8b-instruct
    llama3.1-8b-instruct-lc-rtn-w4a16
    llama3.1-8b-instruct-lc-smooth-rtn-w4a16
    llama3.1-8b-instruct-lc-rtn-w8a16
    nm-llama3.1-8b-instruct-gptq-w4a16
    hf-llama3.1-8b-instruct-awq-4bit

    # # # 2.4 Ministral 8B
    ministral-8b-instruct
    ministral-8b-instruct-lc-rtn-w4a16
    ministral-8b-instruct-lc-smooth-rtn-w4a16
    ministral-8b-instruct-lc-rtn-w8a16
    ministral-8b-instruct-lc-gptq-w4a16
    ministral-8b-instruct-awq-w4a16

    # # # 2.6. Qwen2 7B
    qwen2-7b
    qwen2-7b-instruct
    qwen2-7b-instruct-lc-rtn-w4a16
    qwen2-7b-instruct-lc-smooth-rtn-w4a16
    qwen2-7b-instruct-lc-rtn-w8a16
    hf-qwen2-7b-instruct-awq-w4a16
    hf-qwen2-7b-instruct-gptq-w4a16

    # 2.8. Qwen2.5 0.5B
    qwen2.5-0.5b
    qwen2.5-0.5b-instruct
    qwen2.5-0.5b-instruct-awq-w4a16
    qwen2.5-0.5b-instruct-gptq-w4a16
    qwen2.5-0.5b-instruct-lc-rtn-w4a16
    qwen2.5-0.5b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-0.5b-instruct-lc-rtn-w8a16

    # # # 2.9. Qwen2.5 1.5B
    qwen2.5-1.5b
    qwen2.5-1.5b-instruct
    qwen2.5-1.5b-instruct-awq-w4a16
    qwen2.5-1.5b-instruct-gptq-w4a16
    qwen2.5-1.5b-instruct-lc-rtn-w4a16
    qwen2.5-1.5b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-1.5b-instruct-lc-rtn-w8a16

    # # Qwen2.5 3B
    qwen2.5-3b
    qwen2.5-3b-instruct
    qwen2.5-3b-instruct-awq-w4a16
    qwen2.5-3b-instruct-gptq-w4a16
    qwen2.5-3b-instruct-lc-rtn-w4a16
    qwen2.5-3b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-3b-instruct-lc-rtn-w8a16

    # Qwen2.5 7B
    qwen2.5-7b
    qwen2.5-7b-instruct
    qwen2.5-7b-instruct-awq-w4a16
    qwen2.5-7b-instruct-gptq-w4a16
    qwen2.5-7b-instruct-lc-rtn-w4a16
    qwen2.5-7b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-7b-instruct-lc-rtn-w8a16

    # Qwen2.5 14B
    qwen2.5-14b
    qwen2.5-14b-instruct
    qwen2.5-14b-instruct-awq-w4a16
    qwen2.5-14b-instruct-gptq-w4a16
    qwen2.5-14b-instruct-lc-rtn-w4a16
    qwen2.5-14b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-14b-instruct-lc-rtn-w8a16
)


################################################################################
#                         Generalized Text Evaluation                          #
################################################################################
# Evaluate model
for MODEL_NAME in "${ALL_MODELS[@]}"; do
    python -m scripts.benchmark bias_evaluate ${MODEL_NAME};
done
