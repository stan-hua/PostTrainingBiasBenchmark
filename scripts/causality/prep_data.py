"""
prep_data.py

Description: Prepare preference optimization data and evaluation data for
             causality experiments on BBQ dataset.
"""

# Standard libraries
import ast
import os

# Non-standard libraries
import pandas as pd

# Custom libraries
import config


################################################################################
#                                  Functions                                   #
################################################################################
def get_initial_response_data(overwrite=False):
    """
    Load and filter BBQ response changes data for causality analysis.

    Parameters
    ----------
    overwrite : bool, optional
        If True, overwrite existing data file, by default False

    Note
    ----
    Filter for Qwen 2.5 0.5B RTN W4A16 model and social groups with at least
    500 samples. Select top 3 and bottom 3 social groups by proportion of
    response flipping.
    """
    # Early return, if data already exists
    if not overwrite and os.path.exists(config.CAUSALITY_PATHS["initial_responses"]):
        return pd.read_csv(config.CAUSALITY_PATHS["initial_responses"])

    # Import custom libraries
    from scripts.analysis import load_closed_dataset_entropy_changes

    # Load BBQ response changes for Qwen 2.5 0.5B RTN W4A16
    df_changes = load_closed_dataset_entropy_changes("BBQ")
    model = "qwen2.5-0.5b-instruct-lc-rtn-w4a16"
    df_changes = df_changes[df_changes["model_modified"] == model]

    # Save data subset to CSV
    os.makedirs(os.path.dirname(config.CAUSALITY_PATHS["initial_responses"]), exist_ok=True)
    df_changes.to_csv(config.CAUSALITY_PATHS["initial_responses"], index=False)

    return df_changes


def create_po_dataset():
    """
    Create preference optimization dataset
    """
    df_data = get_initial_response_data()

    # Parse choices to list
    if isinstance(df_data["choices"].iloc[0], str):
        df_data["choices"] = df_data["choices"].map(ast.literal_eval) 

    # Filter out duplicate prompts
    df_data = df_data.drop_duplicates(subset=["prompt"])

    # Create preference pairs by pairing unknown and stereotypical response
    df_data["accept_response"] = df_data.apply(
        lambda row: row["choices"][row["unknown_label"]-1],
        axis=1,
    )
    df_data["reject_response"] = df_data.apply(
        lambda row: row["choices"][row["target_label"]-1],
        axis=1,
    )

    # Quick check that the response options differ in length
    # NOTE: Otherwise, might be selecting a single character
    assert df_data["accept_response"].str.len().nunique() != 1
    assert df_data["reject_response"].str.len().nunique() != 1

    ############################################################################
    #                 Create Train/Test from Social Groups                     #
    ############################################################################
    # Combine into a single DataFrame
    df_summary = pd.DataFrame({
        "num_sizes": df_data.groupby("social_group").size(),
        "prop_flipped": df_data.groupby("social_group")["Flipped"].mean()
    }).reset_index()

    # Get top 3 and bottom 3 social groups by prop_flipped
    # NOTE: Social groups must have at least 300 samples
    df_summary = df_summary[df_summary["num_sizes"] >= 300]
    df_summary = df_summary.sort_values(by="prop_flipped", ascending=False)
    top_3 = df_summary.head(3)["social_group"].tolist()
    bottom_3 = df_summary.tail(3)["social_group"].tolist()
    filter_groups = top_3 + bottom_3

    # Assign all data as unseen test set
    df_data["split"] = "unseen_test"

    # Split prompts 50-50 for each social group into train and test sets
    for social_group in filter_groups:
        df_group = df_data[df_data["social_group"] == social_group]
        df_train = df_group.sample(frac=0.5, random_state=42)
        df_test = df_group.drop(df_train.index)
        df_data.loc[df_train.index, "split"] = "train"
        df_data.loc[df_test.index, "split"] = "test"

    # Store only relevant columns
    cols = [
        "idx", "model_base", "model_modified", "split",
        "social_group", "prompt", "accept_response", "reject_response",
        "choices", "res_probs_base", "res_probs_modified",
        'target_label', 'unknown_label', "normalized_entropy_category", 
    ]
    df_data = df_data[cols]

    # Store training and test set separately
    df_train = df_data[df_data["split"] == "train"]
    df_test = df_data[df_data["split"] == "test"]
    df_unseen_test = df_data[df_data["split"] == "unseen_test"]
    df_train.to_csv(config.CAUSALITY_PATHS["train_set"], index=False)
    df_test.to_csv(config.CAUSALITY_PATHS["test_set"], index=False)
    df_unseen_test.to_csv(config.CAUSALITY_PATHS["unseen_test_set"], index=False)


if __name__ == "__main__":
    from fire import Fire
    Fire()
