#!/bin/bash -l
#SBATCH --job-name=run_spo                    # Job name
#SBATCH --account=fc_chenlab
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH -o slurm/logs/slurm-run_spo-%j.out

# If you want to do it in the terminal,
# salloc --nodes=1 --cpus-per-task=8 --mem=16G
# (command)

################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
# Option 1. Using conda
# conda activate fairbench
# (Recommended) Option 2. Pixi
export TRANSFORMERS_NO_TORCHVISION=1
pixi shell -e simpo

# Set output directory
DIR_OUTPUTS="data/save_data/analysis/causality/outputs"

################################################################################
#                               Fine-Tune on BBQ                               #
################################################################################
# Reduce uncertainty (standard gradient descent)
pixi run -e simpo python -m scripts.causality.run_spo train \
    --gradient_ascent=False

# Increase uncertainty (gradient ascent)
pixi run -e simpo python -m scripts.causality.run_spo train \
    --gradient_ascent=True

################################################################################
#                            Merge LoRA checkpoints                            #
################################################################################
# For each gradient ascent/descent
for grad_type in "ga" "gd"; do
    # For each checkpoint
    for ckpt_idx in "82" "164" "246" "328" "410"; do
        DIR_CURR="$DIR_OUTPUTS/${grad_type}/qwen2.5-0.5b-instruct_${grad_type}"
        LORA_PATH="$DIR_CURR/checkpoint-${ckpt_idx}"
        OUTPUT_PATH="$DIR_CURR-checkpoint-${ckpt_idx}-merged"

        # Merge LoRA adapters into Model
        pixi run -e simpo python -m scripts.causality.run_spo merge \
            --lora_path $LORA_PATH \
            --output_path $OUTPUT_PATH
    done
done


################################################################################
#                                Quantize Model                                #
################################################################################
# For each gradient ascent/descent
for grad_type in "ga" "gd"; do
    # For each checkpoint
    for ckpt_idx in "82" "164" "246" "328" "410"; do
        DIR_CURR="$DIR_OUTPUTS/${grad_type}"
        DIR_MODEL="$DIR_CURR/qwen2.5-0.5b-instruct_${grad_type}-checkpoint-${ckpt_idx}-merged"
        DIR_QUANTIZED="$DIR_CURR/rtn_w4_quantized/"

        # Quantize model at each checkpoint
        pixi run -e quantizer python -m scripts.quant.quantize_model \
            rtn \
            --model_path $DIR_MODEL \
            --save_dir $DIR_QUANTIZED
    done
done
