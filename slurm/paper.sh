#!/bin/bash -l
#SBATCH --job-name=paper                    # Job name
#SBATCH --account=fc_chenlab
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24                 # Number of CPU cores per TASK

# --gres=gpu:NVIDIA_L40S:1
#SBATCH --mem=32GB
#SBATCH -o slurm/logs/slurm-paper-%j.out

# If you want to do it in the terminal,
# salloc --nodes=1 --cpus-per-task=8 --mem=16G
# srun (command)

################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
# micromamba activate fairbench

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

################################################################################
#                                Set Constants                                 #
################################################################################
port=$(shuf -i 6000-9000 -n 1)
echo $port

# System prompt type ("no_sys_prompt", "really_1x", "really_2x", "really_3x", "really_4x")
export SYSTEM_PROMPT_TYPE="no_sys_prompt"


# TODO: Remove
# HuggingFace username
export HF_DATA_USERNAME="stan-hua"

################################################################################
#                          Discriminative Evaluation                           #
################################################################################
CLOSED_DATASET_NAMES=(
    # "CEB-Recognition-S"
    # "CEB-Recognition-T"
    # "CEB-Adult"
    # "CEB-Credit"
    # "CEB-Jigsaw"
    # "BBQ"
    # "BiasLens-Choices"
    # "IAT"
    # "SocialStigmaQA"
    # "StereoSet-Intersentence"

    # The following are deprecated:
    # # "BiasLens-YesNo"
    # # "StereoSet-Intrasentence"
)

for DATASET_NAME in "${CLOSED_DATASET_NAMES[@]}"; do
    pixi run -e analysis \
        python -m scripts.analysis cache_closed_dataset_metrics $DATASET_NAME;
done


# DiscrimEval
# srun python -m scripts.analysis analyze_de

################################################################################
#                            Generative Evaluation                             #
################################################################################
OPEN_DATASET_NAMES=(
    # "CEB-Continuation-S"
    # "CEB-Continuation-T"
    # "CEB-Conversation-S"
    # "CEB-Conversation-T"
    # "FMT10K-IM-S"
    # "FMT10K-IM-T"
    # "BiasLens-GenWhy"

    # The following are deprecated:
    # # "BOLD"
    # # "DoNotAnswer-S"
    # # "DoNotAnswer-T"
)

for DATASET_NAME in "${OPEN_DATASET_NAMES[@]}"; do
    pixi run -e analysis \
        python -m scripts.analysis cache_open_dataset_tests $DATASET_NAME;
done



################################################################################
#                                   Results                                    #
################################################################################
# # Figure 1.
# srun python -m scripts.analysis change_in_agg_metrics

# Supp Table 1.
# srun python -m scripts.analysis change_in_agg_metrics_int8

# # # Table 1.
# srun python -m scripts.analysis change_in_response_flipping

# # # Figure 2
# srun python -m scripts.analysis change_in_uncertainty

# # # Figure 3
# srun python -m scripts.analysis factors_related_to_response_flipping

# # Figure 3c
# srun python -m scripts.analysis change_in_response_by_social_group_bbq

# # Supp Table 2.
# srun python -m scripts.analysis changes_in_model_selection

# # Figure 4.
# srun python -m scripts.analysis change_in_text_patterns

# Figure 5.
# srun python -m scripts.analysis change_in_text_bias

# # Supp. Table.
# srun python -m scripts.analysis change_in_text_bias_fmt10k
# srun python -m scripts.analysis change_in_text_bias_biaslens
