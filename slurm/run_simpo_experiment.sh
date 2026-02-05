#!/bin/bash -l
#SBATCH --job-name=run_spo                    # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --mem=32GB
#SBATCH -o slurm/logs/slurm-run_spo-%j.out

# If you want to do it in the terminal,
# salloc --nodes=1 --cpus-per-task=8 --mem=16G
# (command)

################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
pixi shell -e simpo

# Set output directory
DIR_OUTPUTS="data/save_data/analysis/causality/outputs"
DIR_OUTPUT_MODELS="$DIR_OUTPUTS/models"

# Base Model
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Flag to overwrite existing results
OVERWRITE=False


################################################################################
#                               Fine-Tune on BBQ                               #
################################################################################
# # Reduce uncertainty (standard gradient descent)
pixi run -e simpo python -m scripts.causality.run_spo train \
    --gradient_ascent=False

# Increase uncertainty (gradient ascent)
pixi run -e simpo python -m scripts.causality.run_spo train \
    --gradient_ascent=True


################################################################################
#                            Merge LoRA checkpoints                            #
################################################################################
# For each gradient ascent/descent
for grad_type in "gd" "ga"; do
    # For each checkpoint
    for ckpt_idx in "84" "168" "252" "336" "420"; do
        DIR_CURR="$DIR_OUTPUT_MODELS/${grad_type}/qwen2.5-0.5b-instruct_${grad_type}"
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
# Quantize original model
pixi run -e quantizer python -m scripts.quant.quantize_model \
    rtn \
    --model_path $BASE_MODEL \
    --save_dir $DIR_OUTPUT_MODELS

# For each gradient ascent/descent
for grad_type in "gd" "ga"; do
    # For each checkpoint
    for ckpt_idx in "84" "168" "252" "336" "420"; do
        DIR_CURR="$DIR_OUTPUT_MODELS/${grad_type}"
        DIR_MODEL="$DIR_CURR/qwen2.5-0.5b-instruct_${grad_type}-checkpoint-${ckpt_idx}-merged"
        DIR_QUANTIZED="$DIR_CURR/rtn_w4_quantized/"
        mkdir -p $DIR_QUANTIZED

        # Quantize model at each checkpoint
        pixi run -e quantizer python -m scripts.quant.quantize_model \
            rtn \
            --model_path $DIR_MODEL \
            --save_dir $DIR_QUANTIZED
    done
done


################################################################################
#                             Perform Predictions                              #
################################################################################
# Predict with original model
for split in "train" "test" "unseen_test"; do
    # Oringal model
    python -m scripts.causality.evaluate infer \
        --model_path_or_name $BASE_MODEL \
        --split $split \
        --overwrite=$OVERWRITE

    # Quantized original model
    # NOTE: Quantized model is stored on HuggingFace.
    #       To use local model, change HF_DATA_USERNAME to DIR_OUTPUT_MODELS
    python -m scripts.causality.evaluate infer \
        --model_path_or_name $HF_DATA_USERNAME/Qwen2.5-0.5B-Instruct-LC-RTN-W4A16 \
        --split $split \
        --overwrite=$OVERWRITE
done

# For each gradient ascent/descent
for grad_type in "gd" "ga"; do
    # For each checkpoint
    for ckpt_idx in "84" "168" "252" "336" "420"; do
        for split in "train" "test" "unseen_test"; do
            DIR_CURR="$DIR_OUTPUT_MODELS/${grad_type}"
            DIR_ORIGINAL="$DIR_CURR/qwen2.5-0.5b-instruct_${grad_type}-checkpoint-${ckpt_idx}-merged"
            DIR_QUANTIZED="$DIR_CURR/rtn_w4_quantized/qwen2.5-0.5b-instruct_${grad_type}-checkpoint-${ckpt_idx}-merged-LC-RTN-W4A16"

            # 1. Predict with unquantized checkpoint
            python -m scripts.causality.evaluate infer \
                --model_path_or_name $DIR_ORIGINAL \
                --split $split \
                --overwrite=$OVERWRITE

            # 2. Predict with quantized checkpoint
            python -m scripts.causality.evaluate infer \
                --model_path_or_name $DIR_QUANTIZED \
                --split $split \
                --overwrite=$OVERWRITE
        done
    done
done
