"""
evaluate.py

Description: Evaluate models before and after RTN W4A16 quantization, modulating
             uncertainty using SimPO, on training and held-out test data.
"""

# Standard libraries
import json
import os

# Non-standard libraries
import pandas as pd
from glob import glob

# Custom libraries
import config
from src.utils.llm_gen_wrapper import LLMGeneration, extract_model_path_or_name


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


def evaluate_flipping(split, gradient_ascent=False, overwrite=False):
    """
    Check for response flipping for the data split

    Parameters
    ----------
    split : str
        The data split being evaluated ('train', 'test', 'unseen_test').
    gradient_ascent : bool, optional
        If True, evaluate models that did gradient ascent (more uncertainty).
        Default is False.
    overwrite : bool, optional
        Whether to overwrite existing evaluation results. Default is False.
    """
    infix = "ga" if gradient_ascent else "gd"

    # Get all predictions
    save_path_regex = os.path.join(
        config.CAUSALITY_PATHS["evaluations_dir"], f"*_{infix}_*",
        f"{split}_evaluation.json",
    )

    # Create save path
    save_path = os.path.join(save_dir, f"{split}_evaluation.json")
    os.makedirs(save_dir, exist_ok=True)

    # Early return, if evaluation already exists
    if not overwrite and os.path.exists(save_path):
        print(f"Evaluation already exists at {save_path}. Skipping evaluation.")
        with open(save_path, "r") as f:
            return json.load(f)

    # Load predictions
    pred_path = os.path.join(config.CAUSALITY_PATHS["predictions_dir"], model_name, f"{split}_predictions.csv")
    df_preds = pd.read_csv(pred_path)

    # Load ground truth data
    df_data, _ = load_data(split)

    # Merge predictions with ground truth
    df_merged = pd.merge(df_data, df_preds, on="idx", suffixes=("_true", "_pred"))

    # Calculate accuracy
    correct_preds = df_merged.apply(
        lambda row: row["choices_pred"].index(row["accept_response_true"]) < 
                    row["choices_pred"].index(row["reject_response_true"]),
        axis=1
    )
    accuracy = correct_preds.mean()

    # Save evaluation results
    eval_results = {"accuracy": accuracy}
    with open(save_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    return eval_results



if __name__ == "__main__":
    from fire import Fire
    Fire()
