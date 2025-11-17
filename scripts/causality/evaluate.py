"""
evaluate.py

Description: Evaluate models before and after RTN W4A16 quantization, modulating
             uncertainty using SimPO, on training and held-out test data.
"""

# Standard libraries
import ast
import json
import os

# Non-standard libraries
import numpy as np
import pandas as pd
from glob import glob

# Custom libraries
import config


################################################################################
#                                  Functions                                   #
################################################################################
def load_data(split):
    """
    Load data for evaluation.

    Parameters
    ----------
    split : str
        The data split to load ('train', 'test', 'unseen_test').

    Returns
    -------
    tuple of (pd.DataFrame, list of dict)
        (i) Loaded data in tabular format
        (ii) Data simplified to row format (idx, prompt, choices)
    """
    df_data = pd.read_csv(config.CAUSALITY_PATHS[f"{split}_set"])

    # Convert to row format for LLMGeneration
    data_dict = df_data.apply(
        lambda row: {
            "idx": row["idx"],
            "prompt": row["prompt"],
            "choices": [row["accept_response"], row["reject_response"]],
        },
        axis=1
    ).tolist()

    return df_data, data_dict


def infer(model_path_or_name, split, overwrite=False):
    """
    Perform inference using the specified model on the provided data.

    Parameters
    ----------
    model_path_or_name : str
        Model name, or path to HuggingFace model
    split : str
        The data split being evaluated ('train', 'test', 'unseen_test').
    overwrite : bool, optional
        Whether to overwrite existing predictions. Default is False.
    """
    # Late import
    from src.utils.llm_gen_wrapper import LLMGeneration, extract_model_path_or_name

    # Create save path
    model_name, model_path = extract_model_path_or_name(model_path_or_name)
    save_dir = os.path.join(config.CAUSALITY_PATHS["predictions_dir"], model_name)
    save_path = os.path.join(save_dir, f"{split}_predictions.csv")
    os.makedirs(save_dir, exist_ok=True)

    # Early return, if predictions already exist
    if not overwrite and os.path.exists(save_path):
        print(f"Skipping! Predictions exist for model: `{model_name}`")
        return pd.read_csv(save_path)

    # Load model
    print(f"Performing inference for model: `{model_name}`")
    model_wrapper = LLMGeneration(model_path_or_name=model_path)

    # Prepare data
    df_data, data_list = load_data(split)

    # Predict each row
    accum_data = []
    for row in data_list:
        model_wrapper.process_row_single_turn(
            row,
            index=row["idx"],
            temperature=1,
            key_name="prompt",
        )
        accum_data.append(row)
    df_accum = pd.DataFrame(accum_data)

    # Add columns for model name
    df_accum["model"] = model_wrapper.model_name

    # Save predictions
    df_accum.to_csv(save_path, index=False)

    print(f"Done!")
    return df_accum


def pair_responses(split, overwrite=False):
    """
    Pair all unquantized and quantized model checkpoints

    Parameters
    ----------
    split : str
        The data split being evaluated ('train', 'test', 'unseen_test').
    overwrite : bool, optional
        Whether to overwrite existing evaluation results. Default is False.
    """
    # Create save path
    save_dir = config.CAUSALITY_PATHS["results_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{split}-paired_preds.csv")

    # Early return, if evaluation already exists
    if not overwrite and os.path.exists(save_path):
        print(f"Evaluation already exists at `{save_path}`! Skipping evaluation...")
        return pd.read_csv(save_path)

    # Get all predictions
    save_path_regex = os.path.join(
        config.CAUSALITY_PATHS["predictions_dir"], f"*",
        f"{split}_predictions.csv",
    )

    # Map predictions to before and after quantization
    name_to_base = {}
    name_to_modified = {}
    for pred_path in glob(save_path_regex):
        df_preds = pd.read_csv(os.path.join(pred_path))
        model_name = df_preds["model"].iloc[0]
        base_model_name = get_base_model(model_name)
        # Parse epoch
        if "epoch_" in base_model_name:
            df_preds["epoch"] = int(base_model_name.split("epoch_")[-1].split("-")[0])
        else:
            df_preds["epoch"] = -1
        # Parse fine-tuning
        if "_gd-" in base_model_name:
            df_preds["finetuned"] = "gd"
        elif "_ga-" in base_model_name:
            df_preds["finetuned"] = "ga"
        else:
            df_preds["finetuned"] = "none"
        # Parse probabilities
        df_preds["res_probs"] = df_preds["res_probs"].apply(ast.literal_eval)
        # Parse choices
        df_preds["choices"] = df_preds["choices"].apply(ast.literal_eval)
        # Compute uncertainty (entropy) in unquantized versus quantized
        df_preds["res_probs_entropy"] = df_preds["res_probs"].apply(compute_entropy)
        # Create flag if response is biased
        df_preds["is_biased"] = df_preds.apply(
            lambda row: row["choices"][0] != row["res"],
            axis=1,
        )
        # CASE 1: Unquantized model
        if model_name == base_model_name:
            name_to_base[model_name] = df_preds
        # CASE 2: Quantized model
        else:
            name_to_modified[base_model_name] = df_preds

    # Pair predictions before and after quantization
    all_model_names = set(name_to_base.keys()).intersection(set(name_to_modified.keys()))
    accum_data = []
    for model_name in all_model_names:
        df_base = name_to_base[model_name]
        df_modified = name_to_modified[model_name]
        # Separate out identical columns
        shared_cols = ["prompt", "choices", "finetuned", "epoch"]
        df_shared = df_base[["idx"] + shared_cols]
        df_base = df_base.drop(columns=shared_cols)
        df_modified = df_modified.drop(columns=shared_cols)
        # Merge on idx
        df_merged = pd.merge(
            df_base, df_modified,
            on="idx",
            suffixes=("_base", "_modified"),
        )
        # Merge back shared columns
        df_merged = pd.merge(
            df_merged, df_shared,
            on="idx",
            how="left"
        )
        # Calculate response/bias flipping
        df_merged["response_flipped"] = df_merged["res_base"] != df_merged["res_modified"]
        df_merged["bias_flipped"] = df_merged.apply(
            lambda row:
                None if not row["response_flipped"] else
                (("B" if row["is_biased_base"] else "UnB") + "->" +
                ("B" if row["is_biased_modified"] else "UnB")),
            axis=1
        )
        # Accumulate paired predictions
        accum_data.append(df_merged)

    # Concatenate all model evaluations
    df_paired = pd.concat(accum_data, ignore_index=True)

    # Add social group information
    df_metadata = pd.read_csv(config.CAUSALITY_PATHS[f"{split}_set"])
    df_metadata = df_metadata[["idx", "social_group"]]
    df_paired = pd.merge(
        df_metadata, df_paired,
        how="right",
        on="idx",
    )

    # Create bins for uncertainty
    df_paired["uncertainty_bin_base"] = pd.cut(
        df_paired["res_probs_entropy_base"], 5,
    )

    # Save paired data
    df_paired.to_csv(save_path, index=False)

    # Flipping by social group
    group_cols = ["finetuned", "epoch", "social_group"]
    df_prop = df_paired.groupby(group_cols)["response_flipped"].mean().reset_index()

    # Pivot the table
    pivot_df = df_prop.pivot_table(
        index=["finetuned", "social_group"],
        columns="epoch",
        values="response_flipped"
    )
    pivot_df = pivot_df.rename_axis(columns="epoch").reset_index()

    # Flipping by uncertainty bin
    group_cols = ["finetuned", "epoch", "uncertainty_bin_base"]
    df_prop = df_paired.groupby(group_cols)["response_flipped"].mean().reset_index()
    pivot_df = df_prop.pivot_table(
        index=["finetuned", "uncertainty_bin_base"],
        columns="epoch",
        values="response_flipped"
    )
    pivot_df = pivot_df.rename_axis(columns="epoch").reset_index()

    return df_paired


################################################################################
#                               Helper Functions                               #
################################################################################
def get_base_model(model_name):
    """
    Get name of model pre-quantization

    Parameters
    ----------
    model_name : str
        Full model name

    Returns
    -------
    str
        Base model name
    """
    all_base_models = config.MODEL_INFO["model_group"]
    instruct_models = [m for m in all_base_models if "instruct" in m]
    non_instruct_models = [m for m in all_base_models if "instruct" not in m]

    # Find model among instruct models first then base
    # NOTE: Choose longest matching base model name
    curr_base_model = None
    for base_model in instruct_models + non_instruct_models:
        if base_model in model_name and (curr_base_model is None or len(base_model) > len(curr_base_model)):
            curr_base_model = base_model

    return curr_base_model


def compute_entropy(values, base=2):
    if values is None:
        return None
    values = np.array(values)
    values = values[values > 0]
    values /= values.sum()  # Normalize to make a probability distribution
    return -np.sum(values * np.log(values) / np.log(base))


if __name__ == "__main__":
    from fire import Fire
    Fire()
