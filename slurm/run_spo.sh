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


################################################################################
#                               Fine-Tune on BBQ                               #
################################################################################
# Reduce uncertainty (standard gradient descent)
pixi run -e simpo python -m scripts.causality.run_spo train \
    --gradient_ascent=False \
    --merge

# Increase uncertainty (gradient ascent)
pixi run -e simpo python -m scripts.causality.run_spo train \
    --gradient_ascent=True \
    --merge

################################################################################
#                                Quantize Model                                #
################################################################################
pixi run -e quantizer python -m scripts.quant.quantize_model \
    smooth_rtn \
    --model_path data/save_data/analysis/causality/outputs/qwen2.5-0.5b-instruct_ga_merged/ \