"""
validate_llm_guard.py

Description: Scripts used to validate the use of LLM Guard

Usage:
    `pixi run -e vllm python -m scripts.validation.validate_llm_guard perform_llm_guard_validation --judge_model [llama|qwen]`

Prerequisites:
    1. Run analysis.py / `sample_open_dataset_for_llama_guard_validation`
    2. Annotate and store in `ANNOTATION_PATH` (below)
"""


# Standard libraries
import os
import tempfile
import warnings
from pathlib import Path

# Non-standard libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Custom libraries
import config
from src.utils.text_evaluator import OpenTextEvaluator, extract_llm_guard_response


################################################################################
#                                    Setup                                     #
################################################################################
warnings.filterwarnings('ignore')

# Location to store dataset audits
DIR_DATASET_AUDIT = os.path.join(config.DIR_ANALYSIS, "open_datasets_audit")

# Save path
ANNOTATION_PATH = os.path.join(DIR_DATASET_AUDIT, "biaslens_and_ceb_sample-annotated.csv")
METRICS_PATH = os.path.join(DIR_DATASET_AUDIT, "biaslens_and_ceb_sample-annotation-metrics.csv")

################################################################################
#                               Helper Functions                               #
################################################################################
def analyze_llm_guard_validation(annotations_path=ANNOTATION_PATH, judge_model="qwen"):
    """
    Analyze LLaMA Guard for BiasLens and CEB Continuation/Conversation
    """
    # Load annotations
    df_samples = pd.read_csv(annotations_path)

    # Columns
    llama_base_col = f"eval_{judge_model}-is_safe_base"
    llama_modified_col = f"eval_{judge_model}-is_safe_modified"
    human_base_col = "eval_human-is_safe_base"
    human_modified_col = "eval_human-is_safe_modified"
    low_quality_base_col = "eval_human-is_trash_base"
    low_quality_modified_col = "eval_human-is_trash_modified"

    # Fill missing in low-quality column with False
    df_samples[low_quality_base_col] = df_samples[low_quality_base_col].fillna(False)
    df_samples[low_quality_modified_col] = df_samples[low_quality_modified_col].fillna(False)

    def _compute_metrics_indiv(df_curr, name="All", subset="both"):
        """
        Compute metrics on pre-quantization and post-quantization responses
        independently
        """
        if subset == "both":
            llama_labels = df_curr[llama_base_col].tolist() + df_curr[llama_modified_col].tolist()
            human_labels = df_curr[human_base_col].tolist() + df_curr[human_modified_col].tolist()
        elif subset == "base":
            llama_labels = df_curr[llama_base_col].tolist()
            human_labels = df_curr[human_base_col].tolist()
        elif subset == "modified":
            llama_labels = df_curr[llama_modified_col].tolist()
            human_labels = df_curr[human_modified_col].tolist()
        curr_metrics = compute_metrics(human_labels, llama_labels)
        curr_metrics["name"] = name
        curr_metrics["support"] = len(human_labels)
        return curr_metrics

    def _compute_metrics_paired(df_curr, name="All Paired"):
        """
        Compute metrics on identifying change in pre-quantization and
        post-quantization bias in responses
        """
        llama_shifted = (df_curr[llama_base_col] != df_curr[llama_modified_col]).tolist()
        human_shifted = (df_curr[human_base_col] != df_curr[human_modified_col]).tolist()
        curr_metrics = compute_metrics(human_shifted, llama_shifted)
        curr_metrics["name"] = name
        curr_metrics["support"] = len(human_shifted)
        return curr_metrics

    def _create_confusion_matrix(df_curr):
        """
        Create confusion matrix for pre-quantization and post-quantization
        responses separately.
        """
        cm_base = confusion_matrix(df_curr[human_base_col], df_curr[llama_base_col])
        cm_quantized = confusion_matrix(df_curr[human_modified_col], df_curr[llama_modified_col])
        return cm_base, cm_quantized

    # Compute tpr/tnr/ppv/npv of LLaMA Guard 3 against human annotations
    accum_metrics = []

    # 1. Across all datasets
    accum_metrics.append(_compute_metrics_indiv(df_samples, "All"))
    # accum_metrics.append(_compute_metrics_indiv(df_samples, "All (Baseline)", subset="base"))
    # accum_metrics.append(_compute_metrics_indiv(df_samples, "All (Quantized)", subset="modified"))


    accum_metrics.append(_compute_metrics_paired(df_samples, "All - Paired"))

    # # Decompose to biased -> unbiased
    # mask = (df_samples[human_base_col] & ~df_samples[human_modified_col])
    # mask = mask | (df_samples[human_base_col] == df_samples[human_modified_col])
    # accum_metrics.append(_compute_metrics_paired(df_samples[mask], "All - Paired (B->UnB)"))

    # # Decompose to unbiased -> biased
    # mask = (~df_samples[human_base_col] & df_samples[human_modified_col])
    # mask = mask | (df_samples[human_base_col] == df_samples[human_modified_col])
    # accum_metrics.append(_compute_metrics_paired(df_samples[mask], "All - Paired (UnB->B)"))

    # # Filter on low-quality
    # df_samples_clean = df_samples[~(df_samples[low_quality_base_col] | df_samples[low_quality_modified_col])]
    # accum_metrics.append(_compute_metrics_indiv(df_samples_clean, "All (Filtered)"))
    # accum_metrics.append(_compute_metrics_paired(df_samples_clean, "All - Paired (Filtered)"))

    # 2. Dataset-Specific
    for dataset in df_samples["dataset"].unique().tolist():
        df_dataset = df_samples[df_samples["dataset"] == dataset]
        accum_metrics.append(_compute_metrics_indiv(df_dataset, f"{dataset}"))
        accum_metrics.append(_compute_metrics_paired(df_dataset, f"{dataset} - Paired"))
        # df_dataset_clean = df_dataset[~df_dataset[low_quality_base_col] & ~df_dataset[low_quality_modified_col]]
        # accum_metrics.append(_compute_metrics_indiv(df_dataset_clean, f"{dataset} (Filtered)"))
        # accum_metrics.append(_compute_metrics_paired(df_dataset_clean, f"{dataset} - Paired (Filtered)"))

    df = pd.DataFrame(accum_metrics)
    df.to_csv(METRICS_PATH)


def perform_llm_guard_validation(save_path=ANNOTATION_PATH, judge_model="qwen"):
    """
    Add LLM Guard (LLaMA 3 8B vs. Qwen 3 8B) evaluations to file.

    Parameters
    ----------
    save_path : str, optional
        CSV containing paired questions
    """
    # Load file
    df_paired = pd.read_csv(save_path)

    # Load lazy judge
    llm_judge = OpenTextEvaluator(judge_model)
    judge_response_col = llm_judge.llm_response_col
    prompt_col = "prompt"

    # Iterate over unquantized and quantized responses
    prompts = df_paired[prompt_col].tolist()
    for suffix in ["_base", "_modified"]:
        res_col = f"res{suffix}"
        store_col = f"eval_{judge_model}-is_safe{suffix}"

        # Skip, if eval column already exists and is completely non-empty
        if store_col in df_paired.columns and not df_paired[store_col].isna().any():
            continue

        # Prepare prompt and responses
        curr_responses = df_paired[res_col].tolist()
        accum_data = []
        for idx, curr_response in enumerate(curr_responses):
            curr_data = {"prompt": prompts[idx], "res": curr_response}
            accum_data.append(curr_data)

        # Pass into LLM judge (modify dictionaries in place)
        llm_judge.perform_llm_eval(accum_data)

        # Extract judge response and store
        accum_eval_safe = []
        for curr_data in accum_data:
            curr_eval_safe = extract_llm_guard_response(
                text=curr_data[judge_response_col],
                judge_model=judge_model,
            )[f"eval_{judge_model}-is_safe"]
            accum_eval_safe.append(curr_eval_safe)

        # Store judge responses
        df_paired[store_col] = accum_eval_safe

        # Intermediately store
        atomic_pandas_to_csv(df_paired, save_path)

    return df_paired


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics: tpr, tnr, ppv, and npv.

    Parameters
    ----------
    y_true : array_like
        Ground truth (correct) labels. Should be a 1D list-like or NumPy array of binary values (0 or 1).
    y_pred : array_like
        Predicted labels. Should be a 1D list-like or NumPy array of binary values (0 or 1).

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'tpr' : float
            True Positive Rate (Sensitivity, Recall).
        - 'tnr' : float
            True Negative Rate (Specificity).
        - 'ppv' : float
            Positive Predictive Value (Precision).
        - 'npv' : float
            Negative Predictive Value.

    Notes
    -----
    - tpr = tp / (tp + fn)
    - tnr = tn / (tn + fp)
    - ppv = tp / (tp + fp)
    - npv = tn / (tn + fn)

    Examples
    --------
    >>> y_true = [1, 0, 1, 1, 0, 0, 1]
    >>> y_pred = [1, 0, 1, 0, 0, 1, 1]
    >>> compute_metrics(y_true, y_pred)
    {'tpr': 0.75, 'tnr': 0.666..., 'ppv': 0.75, 'npv': 0.666...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Metrics with safe division
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    # Compute confidence intervals
    n = len(y_true)
    ret = {
        'ppv': ppv, 'npv': npv,
        # 'tpr': tpr, 'tnr': tnr,
    }
    for metric_name, prop in list(ret.items()):
        curr_prop = round(prop, 4)
        curr_prop_pi = [round(float(i), 4) for i in proportion_ci(prop, n)]
        curr_text = f"{curr_prop} {curr_prop_pi}"
        ret[metric_name] = curr_text

    return ret


def proportion_ci(p, n, alpha=0.05):
    """
    Approximate confidence interval for proportion using normal distribution

    Parameters
    ----------
    p : float
        Probability
    n : float
        Number of samples
    alpha : float, optional
        Significance level, by default 0.05

    Returns
    -------
    tuple
        (i) Lower bound
        (ii) Upper bound
    """
    if n == 0 or np.isnan(p):
        return (np.nan, np.nan)
    z = 1.96  # for 95% CI
    se = np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def atomic_pandas_to_csv(
    df: pd.DataFrame,
    save_path: str | os.PathLike,
    **kwargs,
) -> None:
    """
    Atomically write `df` to `save_path` as CSV.
    """
    save_path = Path(save_path)
    directory = save_path.parent

    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        dir=directory,
        prefix=f".{save_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        temp_path = Path(f.name)
        try:
            df.to_csv(f, **kwargs)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    # File is now closed — safe to rename on all platforms.
    try:
        os.replace(temp_path, save_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


################################################################################
#                            Command-line Interface                            #
################################################################################
if __name__ == "__main__":
    from fire import Fire
    Fire()
