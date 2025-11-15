"""
prep_data.py

Description: Prepare preference optimization data and evaluation data for
             causality experiments on BBQ dataset.
"""

# Standard libraries
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
    if os.path.exists(config.CAUSALITY_PATHS["initial_responses"]):
        return pd.read_csv(config.CAUSALITY_PATHS["initial_responses"])

    # Import custom libraries
    from scripts.analysis import load_closed_dataset_entropy_changes

    # Load BBQ response changes for Qwen 2.5 0.5B RTN W4A16
    df_changes = load_closed_dataset_entropy_changes("BBQ")
    model = "qwen2.5-0.5b-instruct-lc-rtn-w4a16"
    df_changes = df_changes[df_changes["model_modified"] == model]

    # Identify response flipping by social group
    num_sizes = df_changes.groupby("social_group").apply(len)
    prop_flipped = df_changes.groupby("social_group")["Flipped"].mean()

    # Combine into a single DataFrame
    df_summary = pd.DataFrame({
        "num_sizes": num_sizes,
        "prop_flipped": prop_flipped
    }).reset_index()

    # Filter for social groups with at least 300 samples
    df_summary = df_summary[df_summary["num_sizes"] >= 300]

    # Get top 3 and bottom 3 social groups by prop_flipped
    df_summary = df_summary.sort_values(by="prop_flipped", ascending=False)
    top_3 = df_summary.head(3)["social_group"].tolist()
    bottom_3 = df_summary.tail(3)["social_group"].tolist()

    # Get data for top 3 and bottom 3 social groups
    df_subset = df_changes[df_changes["social_group"].isin(top_3 + bottom_3)]

    # Save data subset to CSV
    os.makedirs(os.path.dirname(config.CAUSALITY_PATHS["initial_responses"]), exist_ok=True)
    df_subset.to_csv(config.CAUSALITY_PATHS["initial_responses"], index=False)

    return df_subset


def create_po_dataset():
    """
    Create preference optimization dataset
    """
    df_data = get_initial_response_data()

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

    # Split prompts 50-50 for each social group into train and test sets
    for social_group in df_data["social_group"].unique():
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
    df_train.to_csv(config.CAUSALITY_PATHS["train_set"], index=False)
    df_test.to_csv(config.CAUSALITY_PATHS["test_set"], index=False)
