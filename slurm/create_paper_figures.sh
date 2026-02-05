#!/bin/bash -l
#SBATCH --job-name=paper                    # Job name
#SBATCH --time=02:00:00
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # Number of CPU cores per TASK
#SBATCH --mem=16GB
#SBATCH -o slurm/logs/slurm-paper-%j.out

# If you want to do it in the terminal,
# salloc --nodes=1 --cpus-per-task=8 --mem=16G
# (command)

################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
pixi shell -e analysis

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

################################################################################
#                                Set Constants                                 #
################################################################################
port=$(shuf -i 6000-9000 -n 1)
echo $port

# System prompt type
export SYSTEM_PROMPT_TYPE="no_sys_prompt"


################################################################################
#                          Discriminative Evaluation                           #
################################################################################
CLOSED_DATASET_NAMES=(
    "CEB-Recognition-S"
    "CEB-Recognition-T"
    "CEB-Adult"
    "CEB-Credit"
    "CEB-Jigsaw"
    "BBQ"
    "BiasLens-Choices"
    "IAT"
    "SocialStigmaQA"
    "StereoSet-Intersentence"
)

for DATASET_NAME in "${CLOSED_DATASET_NAMES[@]}"; do
    python -m scripts.analysis cache_closed_dataset_metrics $DATASET_NAME;
done


################################################################################
#                            Generative Evaluation                             #
################################################################################
OPEN_DATASET_NAMES=(
    "CEB-Continuation-S"
    "CEB-Continuation-T"
    "CEB-Conversation-S"
    "CEB-Conversation-T"
    "FMT10K-IM-S"
    "FMT10K-IM-T"
    "BiasLens-GenWhy"
)

for DATASET_NAME in "${OPEN_DATASET_NAMES[@]}"; do
    python -m scripts.analysis cache_open_dataset_tests $DATASET_NAME;
done


################################################################################
#                                   Results                                    #
################################################################################

# Define all analysis tasks
tasks=(
    # Figure 2.
    change_in_uncertainty
    # Figure 3
    change_in_agg_metrics
    # Figure 4a&b
    factors_related_to_behavior_flipping
    # Figure 4c
    reranking_changes_due_to_quantization
    # Figure 5
    asymmetric_impact_questions
    asymmetric_impact_bbq
    asymmetric_impact_group_by_dataset

    # Supp Figure.
    change_in_text_patterns
    # Supp Figure.
    change_in_text_bias
    # Supp Table 1.
    change_in_agg_metrics_int8
    # Supp Table.
    change_in_response_flipping
    # Supp. Table.
    change_in_text_bias_fmt10k
    change_in_text_bias_biaslens
)

# Loop through tasks
for task in "${tasks[@]}"; do
  echo "Running $task..."
  pixi run -e analysis \
  python -m scripts.analysis "$task"
done
