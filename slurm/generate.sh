#!/bin/bash -l
#SBATCH --job-name=generate                    # Job name
#SBATCH -o slurm/logs/slurm-generate-%j.out

#SBATCH --account=fc_chenlab
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per TASK
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:A40:1                      # Request one GPU
#SBATCH --qos=a40_gpu3_normal
#SBATCH --mem=32GB
# --tmp=8GB
#SBATCH --time=1:00:00
# --begin=now+10hours

# If you want to do it in the terminal,
# salloc --qos=gpu_deadline_q --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_L40S:1 --cpus-per-task=2 --mem=16G --tmp 8GB
# salloc --qos=gpu_deadline_q --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_H100_NVL:1 --cpus-per-task=2 --mem=16G --tmp 8GB
# salloc --qos=gpu_deadline_q --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_H100_80GB_HBM3:1 --cpus-per-task=1 --mem=16G --tmp 8GB
# salloc --nodes=1 --cpus-per-task=8 --mem=32G
# srun (command)


################################################################################
#                                 Environment                                  #
################################################################################
# # Load CUDA libraries
# module load gcc/12.1.0
# module load cuda/12.4.1

# Load any necessary modules or activate your virtual environment here
# micromamba activate fairbench
# micromamba activate quip

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Configure PyTorch to avoid defragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Override maximum model length (issue for Phi3)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

################################################################################
#                                 Choose Model                                 #
################################################################################
# HuggingFace ID
HF_ID=$HF_DATA_USERNAME

# Models to generate for
MODEL_NAMES=(
    # # # 2.0. LLaMA 3.2 1B model
    # llama3.2-1b
    # hf-llama3.2-1b-aqlm-pv-2bit-2x8
    # llama3.2-1b-instruct
    # llama3.2-1b-instruct-lc-rtn-w4a16
    # llama3.2-1b-instruct-lc-smooth-rtn-w4a16
    # llama3.2-1b-instruct-lc-rtn-w8a8
    # llama3.2-1b-instruct-lc-smooth-rtn-w8a8
    # llama3.2-1b-instruct-lc-rtn-w8a16
    # llama3.2-1b-instruct-lc-gptq-w4a16
    # llama3.2-1b-instruct-lc-smooth-gptq-w4a16
    llama3.2-1b-instruct-awq-w4a16
    # hf-llama3.2-1b-instruct-aqlm-pv-2bit-2x8

    # # # 2.1. LLaMA 3.2 3B model
    # llama3.2-3b
    # hf-llama3.2-3b-aqlm-pv-2bit-2x8
    # llama3.2-3b-instruct
    # llama3.2-3b-instruct-lc-rtn-w4a16
    # llama3.2-3b-instruct-lc-smooth-rtn-w4a16
    # llama3.2-3b-instruct-lc-rtn-w8a8
    # llama3.2-3b-instruct-lc-smooth-rtn-w8a8
    # llama3.2-3b-instruct-lc-rtn-w8a16
    # llama3.2-3b-instruct-lc-gptq-w4a16
    # llama3.2-3b-instruct-lc-smooth-gptq-w4a16
    # llama3.2-3b-instruct-awq-w4a16
    # hf-llama3.2-3b-instruct-aqlm-pv-2bit-2x8

    # # # 2.2. LLaMA 3.1 8B model
    # llama3.1-8b-instruct
    # llama3.1-8b-instruct-lc-rtn-w4a16
    # llama3.1-8b-instruct-lc-smooth-rtn-w4a16
    # llama3.1-8b-instruct-lc-rtn-w8a8
    # llama3.1-8b-instruct-lc-smooth-rtn-w8a8
    # llama3.1-8b-instruct-lc-rtn-w8a16
    # nm-llama3.1-8b-instruct-gptq-w4a16
    # llama3.1-8b-instruct-lc-smooth-gptq-w4a16
    # hf-llama3.1-8b-instruct-awq-4bit
    # hf-llama3.1-8b-instruct-aqlm-pv-2bit-2x8
    # hf-llama3.1-8b-instruct-aqlm-pv-1bit-1x16

    # # # 2.3. LLaMA 3.1 70B model
    # llama3.1-70b
    # llama3.1-70b-instruct
    # llama3.1-70b-instruct-lc-rtn-w4a16
    # llama3.1-70b-instruct-lc-rtn-w4a16kv8
    # llama3.1-70b-instruct-lc-smooth-rtn-w4a16
    # llama3.1-70b-instruct-lc-rtn-w8a8
    # llama3.1-70b-instruct-lc-smooth-rtn-w8a8
    # llama3.1-70b-instruct-lc-rtn-w8a16
    # nm-llama3.1-70b-instruct-gptq-w4a16
    # hf-llama3.1-70b-instruct-awq-int4
    
    # # # Mistral 7B
    # # mistral-v0.3-7b
    # # mistral-v0.3-7b-instruct

    # # # 2.4 Ministral 8B
    # ministral-8b-instruct
    # ministral-8b-instruct-lc-rtn-w4a16
    # ministral-8b-instruct-lc-smooth-rtn-w4a16
    # ministral-8b-instruct-lc-rtn-w8a8
    # ministral-8b-instruct-lc-smooth-rtn-w8a8
    # ministral-8b-instruct-lc-rtn-w8a16
    # ministral-8b-instruct-lc-gptq-w4a16
    # ministral-8b-instruct-lc-smooth-gptq-w4a16
    # ministral-8b-instruct-awq-w4a16

    # # 2.5 Mistral Small 22B
    # mistral-small-22b-instruct
    # mistral-small-22b-instruct-lc-rtn-w4a16
    # mistral-small-22b-instruct-lc-smooth-rtn-w4a16
    # mistral-small-22b-instruct-lc-rtn-w8a8
    # mistral-small-22b-instruct-lc-smooth-rtn-w8a8
    # mistral-small-22b-instruct-lc-rtn-w8a16
    # mistral-small-22b-instruct-lc-gptq-w4a16
    # mistral-small-22b-instruct-lc-smooth-gptq-w4a16
    # mistral-small-22b-instruct-awq-w4a16

    # # # 2.6. Qwen2 7B
    # qwen2-7b
    # qwen2-7b-instruct
    # qwen2-7b-instruct-lc-rtn-w4a16
    # qwen2-7b-instruct-lc-smooth-rtn-w4a16
    # qwen2-7b-instruct-lc-rtn-w8a8
    # qwen2-7b-instruct-lc-smooth-rtn-w8a8
    # qwen2-7b-instruct-lc-rtn-w8a16
    # hf-qwen2-7b-instruct-awq-w4a16
    # hf-qwen2-7b-instruct-gptq-w4a16
    # hf-qwen2-7b-instruct-gptq-w8a16

    # # # 2.7. Qwen2 72B
    # qwen2-72b
    # qwen2-72b-instruct
    # qwen2-72b-instruct-lc-rtn-w4a16
    # qwen2-72b-instruct-lc-smooth-rtn-w4a16
    # qwen2-72b-instruct-lc-rtn-w8a8
    # qwen2-72b-instruct-lc-smooth-rtn-w8a8
    # qwen2-72b-instruct-lc-rtn-w8a16
    # hf-qwen2-72b-instruct-gptq-w4a16
    # hf-qwen2-72b-instruct-awq-w4a16
    # hf-qwen2-72b-instruct-aqlm-pv-2bit-1x16
    # hf-qwen2-72b-instruct-aqlm-pv-1bit-1x16

    # # # 2.8. Qwen2.5 0.5B
    # qwen2.5-0.5b
    # qwen2.5-0.5b-instruct
    # qwen2.5-0.5b-instruct-awq-w4a16
    # qwen2.5-0.5b-instruct-gptq-w4a16
    # qwen2.5-0.5b-instruct-gptq-w8a16
    # qwen2.5-0.5b-instruct-lc-rtn-w4a16
    # qwen2.5-0.5b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-0.5b-instruct-lc-rtn-w8a8
    # qwen2.5-0.5b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-0.5b-instruct-lc-rtn-w8a16
    # qwen2.5-0.5b-instruct-lc-smooth-rtn-w8a16

    # # # 2.9. Qwen2.5 1.5B
    # qwen2.5-1.5b
    # qwen2.5-1.5b-instruct
    # qwen2.5-1.5b-instruct-awq-w4a16
    # qwen2.5-1.5b-instruct-gptq-w4a16
    # qwen2.5-1.5b-instruct-gptq-w8a16
    # qwen2.5-1.5b-instruct-lc-rtn-w4a16
    # qwen2.5-1.5b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-1.5b-instruct-lc-rtn-w8a8
    # qwen2.5-1.5b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-1.5b-instruct-lc-rtn-w8a16
    # qwen2.5-1.5b-instruct-lc-smooth-rtn-w8a16

    # # # Qwen2.5 3B
    # qwen2.5-3b
    # qwen2.5-3b-instruct
    # qwen2.5-3b-instruct-awq-w4a16
    # qwen2.5-3b-instruct-gptq-w4a16
    # qwen2.5-3b-instruct-gptq-w8a16
    # qwen2.5-3b-instruct-lc-rtn-w4a16
    # qwen2.5-3b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-3b-instruct-lc-rtn-w8a8
    # qwen2.5-3b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-3b-instruct-lc-rtn-w8a16
    # qwen2.5-3b-instruct-lc-smooth-rtn-w8a16

    # # # # Qwen2.5 7B
    # qwen2.5-7b
    # qwen2.5-7b-instruct
    # qwen2.5-7b-instruct-awq-w4a16
    # qwen2.5-7b-instruct-gptq-w4a16
    # qwen2.5-7b-instruct-gptq-w8a16
    # qwen2.5-7b-instruct-lc-rtn-w4a16
    # qwen2.5-7b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-7b-instruct-lc-rtn-w8a8
    # qwen2.5-7b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-7b-instruct-lc-rtn-w8a16
    # qwen2.5-7b-instruct-lc-smooth-rtn-w8a16

    # # # # Qwen2.5 14B
    # qwen2.5-14b
    # qwen2.5-14b-instruct
    # qwen2.5-14b-instruct-awq-w4a16
    # qwen2.5-14b-instruct-gptq-w4a16
    # qwen2.5-14b-instruct-gptq-w8a16
    # qwen2.5-14b-instruct-lc-rtn-w4a16
    # qwen2.5-14b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-14b-instruct-lc-rtn-w8a8
    # qwen2.5-14b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-14b-instruct-lc-rtn-w8a16
    # qwen2.5-14b-instruct-lc-smooth-rtn-w8a16

    # # # Qwen2.5 32B
    # qwen2.5-32b
    # qwen2.5-32b-instruct
    # qwen2.5-32b-instruct-awq-w4a16
    # qwen2.5-32b-instruct-gptq-w4a16
    # qwen2.5-32b-instruct-gptq-w8a16
    # qwen2.5-32b-instruct-lc-rtn-w4a16
    # qwen2.5-32b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-32b-instruct-lc-rtn-w8a8
    # qwen2.5-32b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-32b-instruct-lc-rtn-w8a16
    # qwen2.5-32b-instruct-lc-smooth-rtn-w8a16

    # # # Qwen2.5 72B
    # qwen2.5-72b
    # qwen2.5-72b-instruct
    # qwen2.5-72b-instruct-awq-w4a16
    # qwen2.5-72b-instruct-gptq-w4a16
    # qwen2.5-72b-instruct-gptq-w8a16
    # qwen2.5-72b-instruct-lc-rtn-w4a16
    # qwen2.5-72b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-72b-instruct-lc-rtn-w8a8
    # qwen2.5-72b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-72b-instruct-lc-rtn-w8a16
    # qwen2.5-72b-instruct-lc-smooth-rtn-w8a16

    # # # Phi3 8B
    # phi3-3.8b-instruct
    # phi3-3.8b-instruct-lc-rtn-w4a16
    # phi3-3.8b-instruct-lc-rtn-w8a16
    # phi3-3.8b-instruct-lc-rtn-w8a8
    # phi3-3.8b-instruct-lc-smooth-rtn-w4a16
    # phi3-3.8b-instruct-lc-smooth-rtn-w8a16
    # phi3-3.8b-instruct-lc-smooth-rtn-w8a8

    # # # Phi3 7B
    # phi3-7b-instruct
    # phi3-7b-instruct-lc-rtn-w4a16
    # phi3-7b-instruct-lc-rtn-w8a16
    # phi3-7b-instruct-lc-rtn-w8a8
    # phi3-7b-instruct-lc-smooth-rtn-w4a16
    # phi3-7b-instruct-lc-smooth-rtn-w8a16
    # phi3-7b-instruct-lc-smooth-rtn-w8a8

    # # # Phi3 14B
    # phi3-14b-instruct
    # phi3-14b-instruct-lc-rtn-w4a16
    # phi3-14b-instruct-lc-rtn-w8a16
    # phi3-14b-instruct-lc-rtn-w8a8
    # phi3-14b-instruct-lc-smooth-rtn-w4a16
    # phi3-14b-instruct-lc-smooth-rtn-w8a16
    # phi3-14b-instruct-lc-smooth-rtn-w8a8

    # # Gemma 2B
    # gemma2-2b-instruct
    # gemma2-2b-instruct-lc-rtn-w4a16
    # gemma2-2b-instruct-lc-rtn-w8a16
    # gemma2-2b-instruct-lc-rtn-w8a8
    # gemma2-2b-instruct-lc-smooth-rtn-w4a16
    # gemma2-2b-instruct-lc-smooth-rtn-w8a16
    # gemma2-2b-instruct-lc-smooth-rtn-w8a8

    # # Gemma 9B
    # gemma2-9b-instruct
    # gemma2-9b-instruct-lc-rtn-w4a16
    # gemma2-9b-instruct-lc-rtn-w8a16
    # gemma2-9b-instruct-lc-rtn-w8a8
    # gemma2-9b-instruct-lc-smooth-rtn-w4a16
    # gemma2-9b-instruct-lc-smooth-rtn-w8a16
    # gemma2-9b-instruct-lc-smooth-rtn-w8a8

    # # Gemma 27B
    # gemma2-27b-instruct
    # gemma2-27b-instruct-lc-rtn-w4a16
    # gemma2-27b-instruct-lc-rtn-w8a16
    # gemma2-27b-instruct-lc-rtn-w8a8
    # gemma2-27b-instruct-lc-smooth-rtn-w4a16
    # gemma2-27b-instruct-lc-smooth-rtn-w8a16
    # gemma2-27b-instruct-lc-smooth-rtn-w8a8
)

QUIP_MODELS=(
    # "relaxml/Llama-2-70b-chat-E8P-2Bit"
)

# List of VPTQ models to infer
VPTQ_MODELS=(
    # "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-65536-woft"
    # "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft"
    # "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k16384-0-woft"
)

# Flag to use chat template (include both to run w/ and w/o chat template)
CHAT_FLAGS=(
    # "False"
    "True"
)

# Number of GPUS
NUM_GPUS=1

# Datasets to infer on
DATASET_NAME="all_closed"

# System prompt type ("no_sys_prompt", "really_1x", "really_2x", "really_3x", "really_4x")
export SYSTEM_PROMPT_TYPE="no_sys_prompt"


################################################################################
#                              Perform Benchmark                               #
################################################################################
# Assign port
port=$(shuf -i 6000-9000 -n 1)
echo $port

# 1. Regular models
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for CHAT_FLAG in "${CHAT_FLAGS[@]}"; do
        pixi run -e vllm \
        python -m scripts.benchmark generate ${MODEL_NAME} \
            --use_chat_template $CHAT_FLAG \
            --num_gpus $NUM_GPUS \
            --dataset_name $DATASET_NAME;
    done
done

# # 2. QuIP# models
# for MODEL_NAME in "${QUIP_MODELS[@]}"; do
#     python -m scripts.benchmark generate ${MODEL_NAME};
# done

# # 3. VPTQ models
# for MODEL_NAME in "${VPTQ_MODELS[@]}"; do
#     python -m scripts.benchmark generate ${MODEL_NAME} --model_provider "vptq";
# done