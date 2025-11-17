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
        df_preds = pd.read_csv(save_path)
        # Parse probabilities
        df_preds["res_probs_base"] = df_preds["res_probs_base"].apply(ast.literal_eval)
        df_preds["res_probs_modified"] = df_preds["res_probs_modified"].apply(ast.literal_eval)
        # Parse choices
        df_preds["choices"] = df_preds["choices"].apply(ast.literal_eval)
        return df_preds

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
            df_preds["strategy"] = "gd"
        elif "_ga-" in base_model_name:
            df_preds["strategy"] = "ga"
        else:
            df_preds["strategy"] = "none"
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
        shared_cols = ["prompt", "choices", "strategy", "epoch"]
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

    return df_paired


def investigate_flipping():
    """
    Check how response flipping distributions were affected by SimPO, followed
    by quantization.
    """
    # Get paired (pre/post-quantization) responses for before and after SimPO
    df_paired = pair_responses("unseen_test")
    df_paired = df_paired[df_paired["epoch"].isin([-1, 5])]

    # Compute entropy before/after SimPO for unquantized model
    df_entropy_base = compute_agg_by_group(
        df_paired,
        group_col="social_group",
        value_col="res_probs_entropy_base",
    ).rename(columns={
        "before": "entropy-pre_SimPO-pre_RTN",
        "after": "entropy-post_SimPO-pre_RTN",
    })

    # Compute entropy before/after SimPO for quantized model
    df_entropy_modified = compute_agg_by_group(
        df_paired,
        group_col="social_group",
        value_col="res_probs_entropy_modified",
    ).rename(columns={
        "before": "entropy-pre_SimPO-post_RTN",
        "after": "entropy-post_SimPO-post_RTN",
    })

    # Merge entropy dataframes
    df_entropy = pd.merge(
        df_entropy_base, df_entropy_modified,
        on=["social_group", "strategy"]
    ).set_index(["social_group", "strategy"]).round(3)

    # Compute entropy before/after SimPO for quantized model
    # NOTE: Observation. Groups that weren't flipping much before SimPO
    #       actually flipped more post-SimPO + quantization!
    df_flipping_by_group = compute_agg_by_group(
        df_paired,
        group_col="social_group",
        value_col="response_flipped",
    ).rename(columns={
        "before": "flipping-pre_SimPO",
        "after": "flipping-post_SimPO",
    }).set_index(["social_group", "strategy"]).round(3)

    # Compute entropy before/after SimPO for quantized model
    df_flipping_by_uncertainty = compute_agg_by_group(
        df_paired,
        group_col="uncertainty_bin_base",
        value_col="response_flipped",
    ).rename(columns={
        "before": "flipping-pre_SimPO",
        "after": "flipping-post_SimPO",
    }).set_index(["uncertainty_bin_base", "strategy"]).round(3)


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


def compute_agg_by_group(
        df_paired,
        group_col="social_group",
        value_col="response_flipped",
        agg_func="mean",
    ):
    """
    Aggregate some value column by group

    Parameters
    ----------
    df_paired : pd.DataFrame
        Table of paired responses (before/after quantization)
    group_col : str, optional
        Column to group by. Default is "social_group".
    value_col : str, optional
        Column to aggregate. Default is "response_flipped".
    agg_func : str, optional
        Aggregation function to use. Default is "mean".

    Note
    ----
    Only considers the baseline (none) and finetuned (ga/gd) models at the last
    epoch (5) versus before training (-1).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame
    """
    # Check average change in by group column
    group_cols = ["strategy", "epoch", group_col]
    df_agg = df_paired.groupby(group_cols)[value_col].agg(agg_func).reset_index()
    df_pivot = df_agg.pivot_table(
        index=["strategy", group_col],
        columns="epoch",
        values=value_col
    )
    df_pivot = df_pivot.rename_axis(columns="epoch").reset_index()

    # Split into baseline (none) and finetuned (ga/gd)
    df_baseline = df_pivot[df_pivot["strategy"]=="none"].drop(columns=["strategy", 5])
    df_baseline = df_baseline.rename(columns={-1: "before"})
    df_finetuned = df_pivot[df_pivot["strategy"].isin(["ga","gd"])].drop(columns=[-1])
    df_finetuned = df_finetuned.rename(columns={5: "after"})

    # Merge baseline with finetuned on group
    # NOTE: This is so the agg. metric column for the original model +
    #       SimPO-tuned model are side by side
    df_merged = df_baseline.merge(df_finetuned, on=group_col)

    # Reorder columns
    df_merged = df_merged[[group_col, "strategy", "before", "after"]]

    # Add difference column
    df_merged["diff"] = df_merged["after"] - df_merged["before"]

    return df_merged


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    from fire import Fire
    Fire()
