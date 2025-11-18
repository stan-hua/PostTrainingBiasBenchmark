"""
benchmark.py

Description: Contains high-level functions for benchmarking:

Examples:
1. Perform inference on datasets for a specific model
>>> python -m scripts.benchmark generate [model]

2. Evaluate open-ended text generations
>>> python -m scripts.benchmark bias_evaluate [model]

3. Find unfinished runs
>>> python -m scripts.benchmark find_unfinished --help

4. Delete specific runs
>>> python -m scripts.benchmark delete --help
"""

# Standard libraries
import json
import multiprocessing
import logging
import os
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from glob import glob

# Non-standard libraries
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

# Custom libraries
import config
from src.utils import json_utils


################################################################################
#                                    Setup                                     #
################################################################################
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s : %(levelname)s : %(message)s",
)


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Get system prompt type (during generation)
SYSTEM_PROMPT_TYPE = os.environ.get("SYSTEM_PROMPT_TYPE", "no_sys_prompt")

# Constant to force parallel evaluation
# NOTE: Overrides single worker for local Judge LLM eval
FORCE_PARALLEL = str(os.environ.get("FORCE_PARALLEL", "1")) == 1


################################################################################
#                                  Functions                                   #
################################################################################
def generate(
        model_path_or_name,
        dataset_name="all",
        model_provider="vllm",
        use_chat_template=False,
        num_gpus=None,
    ):
    """
    Generate LLM responses for specific or all evaluation datasets.

    Parameters
    ----------
    model_path_or_name : str
        Path to model, or model (nick)name in `config.py`
    dataset_name : str
        Name of the dataset. If not specififed or "all", generate for all
        datasets.
    model_provider : str
        One of local hosting: ("vllm", "huggingface", "vptq"), or one of online
        hosting: ("deepinfra", "replicate", "other")
    use_chat_template : str
        If True, use chat template for local models
    num_gpus : int
        Optional explicit number of GPUs to use
    """
    # Late import to prevent slowdown
    from src.utils.llm_gen_wrapper import LLMGeneration

    # Shared keyword arguments
    shared_kwargs = {
        # Provided arguments
        "model_path_or_name": model_path_or_name,
        "dataset_name": dataset_name,
        "model_provider": model_provider,
        "use_chat_template": use_chat_template,
        "system_prompt_type": SYSTEM_PROMPT_TYPE,
        # Default arguments
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "debug": False,
    }

    # Add number of GPUs if available
    if num_gpus is not None:
        shared_kwargs["num_gpus"] = num_gpus
    elif torch.cuda.is_available():
        shared_kwargs["num_gpus"] = min(torch.cuda.device_count(), 4)

    # Instantiate LLMGeneration wrapper
    llm_gen = LLMGeneration(**shared_kwargs)

    # Perform inference
    LOGGER.info(f"[Generate] Performing inference for {model_path_or_name}...")
    llm_gen.infer_dataset()
    LOGGER.info(f"[Generate] Performing inference for {model_path_or_name}...DONE")


def find_unfinished(pattern="*", filter_models=None, generation=False, evaluation=False):
    """
    Find all models, matching pattern, who are unfinished with inference

    Parameters
    ----------
    pattern : str
        If `generation`, pattern to identify model result directories.
        If `evaluation`, pattern containing the evaluator choice and optionally
        the model names
    filter_models : list
        List of model names to check for explicitly, if already generated/evaluated
    generation : bool
        If True, check generations, based on pattern.
    evaluation : bool
        If True, check evaluation, based on model list.
    """
    model_to_missing_results = defaultdict(list)
    filter_models = filter_models or []
    if filter_models:
        LOGGER.info(f"[Find Unfinished] Filtering for the following models: \n\t{filter_models}")

    # 1. Generation
    if generation:
        LOGGER.info("[Find Unfinished] Finding models with unfinished generations")
        # Iterate over model directories
        for result_dir in tqdm(glob(os.path.join(config.DIR_GENERATIONS, pattern))):
            model_name = os.path.basename(result_dir)
            # Skip, if filtering strictly
            if filter_models and model_name not in filter_models:
                continue

            # Check each dataset
            for dataset_name in config.COLLECTION_TO_DATASETS["all"]:
                json_paths = glob(os.path.join(result_dir, dataset_name, "*.json"))

                # Early return if missing JSON files
                if not json_paths:
                    model_to_missing_results[model_name].extend([f"{dataset_name} / *.json"])
                    break

                # Check if any of the `res` are missing
                for json_path in json_paths:
                    # Load json
                    infer_data = json_utils.load_json(json_path)
                    # Check if any of the `res` are missing
                    if any("res" not in row for row in infer_data):
                        model_to_missing_results[model_name].append(
                            f"{dataset_name}/{os.path.basename(json_path)}"
                        )
                        LOGGER.error(f"[Benchmark] Missing results for: {os.path.join(model_name, dataset_name, os.path.basename(json_path))}")

        # Log all incomplete models
        if model_to_missing_results:
            LOGGER.error(
                "[Find Unfinished - Generation] The following models are incomplete:"
                "\n" + json.dumps(dict(sorted(model_to_missing_results.items())), indent=4)
            )


    # 2. Evaluation
    if evaluation:
        LOGGER.info("[Find Unfinished - Evaluations] Finding models with unfinished evaluations")
        model_dirs = glob(os.path.join(config.DIR_EVALUATIONS, "*", "*"))
        # Filter models based on pattern
        if pattern != "*":
            model_dirs = [d for idx, d in enumerate(model_dirs) if re.match(pattern, d)]

        # If specified, filter for specific models
        if filter_models:
            assert filter_models, "[Find Unfinished - Evaluations] Please provide list of models via `filter_models`, if searching through evaluations!"
            filter_models = list(set(filter_models))
            is_valid = [model_name in set(filter_models) for model_name in model_names]
            model_dirs = [
                model_dir
                for idx, model_dir in enumerate(model_dirs)
                if is_valid[idx]
            ]

        # Datasets and social axes to check
        indirect_datasets = [f"CEB-{test}-{bias}" for test in ["Continuation", "Conversation"] for bias in ["S", "T"]]
        social_axes = ["age", "gender", "race", "religion"]

        # Get all missing directories
        model_names = [os.path.basename(path) for path in model_dirs]

        # Check each model directory for those missing evals
        LOGGER.info(f"[Find Unfinished - Evaluations] Checking the following models: \n\t{model_names}")
        for idx, curr_model_dir in tqdm(enumerate(model_dirs)):
            curr_model_name = model_names[idx]
            # Check if eval for each dataset is present
            for dataset_name in indirect_datasets:
                curr_dataset_dir = os.path.join(curr_model_dir, dataset_name)
                if not os.path.exists(curr_dataset_dir):
                    model_to_missing_results[curr_model_name].extend([f"{dataset_name} / *.json"])
                    break
                # Check if eval for each dataset / social axis is present
                for social_axis in social_axes:
                    curr_axis_dir = os.path.join(curr_dataset_dir, social_axis)
                    if not os.path.exists(curr_axis_dir) or not os.listdir(curr_axis_dir):
                        model_to_missing_results[curr_model_name].extend([f"{dataset_name} / {social_axis}.json"])

        # Log all incomplete models
        if model_to_missing_results:
            LOGGER.error(
                "[Find Unfinished - Evaluation] The following models are incomplete:"
                "\n" + json.dumps(dict(sorted(model_to_missing_results.items())), indent=4)
            )


def delete(
        model_regex="*", dataset_regex="*", social_regex="*", file_regex="*",
        evaluator_choice="*",
        inference=False,
        evaluation=False,
    ):
    """
    Delete inference and evaluation results for all models for the following
    dataset.

    Note
    ----
    Used when the benchmark has changed.

    Parameters
    ----------
    model_regex : str
        Regex that matches model name in saved LLM generations folder
    dataset_regex : str
        Regex that matches dataset
    social_regex : str
        Regex that matches social axis (e.g., race, religion, gender, age) or "all"
    file_regex : str
        Regex that matches a specific filename
    evaluator_choice : str
        Evaluator choice
    inference : bool
        If True, delete inference results (produced by LLMs)
    evaluation : bool
        If True, delete intermediate evaluation files (from Perspective/ChatGPT)
    """
    assert inference or evaluation, "At least one of `inference` or `evaluation` must be True"

    # 1. Remove all generations
    if inference:
        regex_suffix = f"{model_regex}/*/{dataset_regex}/{file_regex}"
        print("[Delete] Deleting inference results matching following regex: ", regex_suffix)
        time.sleep(3)
        for infer_file in tqdm(glob(config.DIR_GENERATIONS + "/" + regex_suffix)):
            if os.path.isdir(infer_file):
                shutil.rmtree(infer_file)
            else:
                os.remove(infer_file)

    # 2. Remove all saved evaluations
    if evaluation:
        regex_suffix = f"{evaluator_choice}/{model_regex}/{dataset_regex}/{social_regex}/{file_regex}"
        print("[Delete] Deleting evaluation results matching following regex: ", regex_suffix)
        time.sleep(3)
        for eval_file in tqdm(glob(config.DIR_EVALUATIONS + "/" + regex_suffix)):
            if os.path.isdir(eval_file):
                shutil.rmtree(eval_file)
            else:
                os.remove(eval_file)


################################################################################
#                   Dataset / Social Axis - Level Processing                   #
################################################################################
def bias_eval_dataset_collection(model_name, collection_name="all_open", **eval_kwargs):
    """
    Perform bias evaluation for a model on all datasets in the collection

    Parameters
    ----------
    model_name : str
        Name of model
    collection_name : str, optional
        Name of collection, by default "all"
    **eval_kwargs : Any
        Keyword arguments for `OpenTextEvaluator.evaluate`
    """
    dataset_names = config.COLLECTION_TO_DATASETS[collection_name]
    for dataset_name in dataset_names:
        try:
            bias_eval_dataset(model_name, dataset_name, **eval_kwargs)
        except:
            LOGGER.error(f"[{dataset_name}] Failed to perform bias evaluation with error:")
            LOGGER.error(traceback.format_exc())


def bias_eval_dataset(model_name, dataset_name, system_prompt_type=SYSTEM_PROMPT_TYPE, **eval_kwargs):
    """
    Perform evaluation on an open-ended dataset for bias

    Parameters
    ----------
    model_name : str
        Name of model (defined in `config.py`)
    dataset_name : str
        Name of dataset
    system_prompt_type : str
        System prompt type
    **eval_kwargs : Any
        Keyword arguments for `OpenTextEvaluator.evaluate`
    """
    # Ensure dataset is open-ended
    if dataset_name not in config.COLLECTION_TO_DATASETS["all_open"]:
        raise RuntimeError(f"Model `{model_name}` / Dataset `{dataset_name}` is not an open-ended dataset! No need for bias evaluation.")

    # Specify path to model generation directory
    dir_model_gen = os.path.join(config.DIR_GENERATIONS, system_prompt_type, model_name)
    dir_dataset = os.path.join(dir_model_gen, dataset_name)

    # Ensure dataset exists
    if not os.path.exists(dir_dataset):
        raise RuntimeError(f"Model `{model_name}` / Dataset `{dataset_name}` results are not yet generated!")

    # Ensure files exist
    exist_files = os.listdir(dir_dataset)
    expected_files = get_expected_dataset_files(dataset_name)
    missing_files = sorted(set(expected_files) - set(exist_files))
    if missing_files:
        raise RuntimeError(
            f"Model `{model_name}` / Dataset `{dataset_name}` results are not "
            f"yet generated! Missing files: \n\t{missing_files}"
        )

    # Evaluate each JSON one by one
    LOGGER.info(f"Evaluating Model: `{model_name}` | Dataset: `{dataset_name}`...")
    for json_path in glob(os.path.join(dir_dataset, "*.json")):
        bias_process_json(dataset_name, json_path, **eval_kwargs)
    LOGGER.info(f"Evaluating Model: `{model_name}` | Dataset: `{dataset_name}`...DONE")


def bias_process_json(dataset_name, json_path, **eval_kwargs):
    """
    Evaluate the following dataset for bias across all prompts and
    social axes.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    json_path : str
        Path to the JSON file containing the prompt information
    **eval_kwargs : Any
        Keyword arguments for `OpenTextEvaluator.evaluate`
    """
    # Late import
    from src.utils.text_evaluator import OpenTextEvaluator

    social_axis = extract_social_axis(json_path)
    LOGGER.info(f"[`{dataset_name}`] Evaluating `{social_axis}`...")

    # Determine prompt column
    judge_kwargs = {}
    if dataset_name.startswith("FMT10K"):
        judge_kwargs["prompt_col"] = "4-turn Conv"
        judge_kwargs["llm_input_col"] = "4-turn Conv Response"
    judge_kwargs.update(eval_kwargs)

    # Raise error, if file doesn't exist
    if not os.path.exists(json_path):
        raise RuntimeError(f"Generation JSON file `{json_path}` does not exist!")

    # Load inferred data
    infer_data = json_utils.load_json(json_path)

    # Evaluate for specific stereotype
    evaluator = OpenTextEvaluator(
        save_dir=os.path.dirname(json_path),
        save_fname=os.path.basename(json_path),
    )
    try:
        evaluator.evaluate(infer_data, **judge_kwargs)
    except Exception as error_msg:
        LOGGER.info(f"Error occurred while adding evaluation metrics to Dataset: {dataset_name}\n\tError: {error_msg}")
        LOGGER.error(traceback.format_exc())
        return None

    LOGGER.info(f"[`{dataset_name}`] Evaluating `{social_axis}`...DONE")


################################################################################
#                               Helper Functions                               #
################################################################################
def extract_social_axis(json_path):
    # NOTE: Assumes path is in the LLM generation directory
    return ".".join(os.path.basename(json_path).split(".")[:-1])


# NOTE: The following is to do with custom naming conventions
def extract_model_metadata_from_name(model_name):
    """
    Extract metadata from custom model name.

    Note
    ----
    The model name must follow the arbitrary naming convention as seen in
    `config.py`

    Parameters
    ----------
    model_name : str
        Model name

    Returns
    -------
    accum_metadata : dict
        Dictionary containing metadata about the model, including:
            - `w_bits`: The number of bits used for weights
            - `a_bits`: The number of bits used for activations
            - `instruct_tuned`: Whether the model is an instruct model
            - `param_size`: The parameter size of the model (in B)
    """
    accum_metadata = {}

    # 1. Get the number of bits for weights
    regexes = [r"(\d)bit", r"w(\d)a\d*"]
    accum_metadata["w_bits"] = 16
    for regex_str in regexes:
        match_obj = re.search(regex_str, model_name)
        if match_obj:
            accum_metadata["w_bits"] = int(match_obj.group(1))
            break

    # 2. Get the number of bits for activations
    accum_metadata["a_bits"] = 16
    match_obj = re.search(r"w(\d)a(\d*)", model_name)
    if match_obj:
        accum_metadata["a_bits"] = int(match_obj.group(2))

    # 3. Get quantization strategy
    accum_metadata["q_method"] = None
    for q_method in ["rtn", "gptq", "awq", "aqlm"]:
        if q_method in model_name:
            accum_metadata["q_method"] = q_method
    if accum_metadata["q_method"] == "awq":
        accum_metadata["w_bits"] = 4
    # Add smoothquant
    accum_metadata["smoothquant"] = False
    if "-smooth-" in model_name:
        accum_metadata["smoothquant"] = True
    # 3.1. Add full quantization method
    accum_metadata["q_method_bits"] = None
    accum_metadata["q_method_full"] = "Native"
    if accum_metadata["q_method"]:
        accum_metadata["q_method_bits"] = "W" + str(accum_metadata["w_bits"]) + "A" + str(accum_metadata["a_bits"])
        accum_metadata["q_method_full"] = accum_metadata["q_method"].upper() + " " + accum_metadata["q_method_bits"] + (" + SQ" if accum_metadata["smoothquant"] else "")

    # 4. Check if the model is an instruct vs. non-instruct model
    accum_metadata["instruct_tuned"] = "instruct" in model_name

    # 5. Get parameter size (in B)
    match_obj = re.search(r"-(\d*\.?\d*)b-?", model_name)
    assert match_obj, f"[Extract Model Metadata] Failed to extract param_size from model name: {model_name}"
    accum_metadata["param_size"] = float(match_obj.group(1))
    accum_metadata["Model Size (GB)"] = accum_metadata["param_size"] * accum_metadata["w_bits"] / 8

    # 6. Get base model
    all_base_models = config.MODEL_INFO["model_group"]
    instruct_models = [m for m in all_base_models if "instruct" in m]
    non_instruct_models = [m for m in all_base_models if "instruct" not in m]
    accum_metadata["base_model"] = None

    # Find model among instruct models first then base
    # NOTE: Choose longest matching base model name
    curr_base_model = None
    for base_model in instruct_models + non_instruct_models:
        if base_model in model_name and (curr_base_model is None or len(base_model) > len(curr_base_model)):
            accum_metadata["base_model"] = curr_base_model = base_model
    assert curr_base_model is not None, f"[Extract Model Metadata] Failed to find base model for: {model_name}!"

    # Get model family
    accum_metadata["model_family"] = accum_metadata["base_model"].split("-")[0]
    return accum_metadata


def extract_model_path_or_name(model_path_or_name, model_provider="vllm", use_chat_template=False):
    """
    Return tuple of model (nick)name and model path, provided either

    Parameters
    ----------
    model_path_or_name : str
        Path to the model, or model (nick)name
    model_provider : str
        Model provider name
    use_chat_template : bool
        If True, use chat template

    Returns
    -------
    tuple of (str, str)
        (i) Model (nick)name
        (ii) Path to model
    """
    # Get model name and path
    model_path_to_name = MODEL_INFO["model_path_to_name"]
    model_name_to_path = {v:k for k,v in model_path_to_name.items()}
    if model_path_or_name in model_path_to_name:
        model_path = model_path_or_name
        model_name = model_path_to_name[model_path_or_name]
    if model_path_or_name in model_name_to_path:
        model_name = model_path_or_name
        model_path = model_name_to_path[model_path_or_name]
    elif model_path_or_name.split("/")[-1] in model_path_to_name:
        model_path = model_path_or_name
        model_name = model_path_to_name[model_path_or_name.split("/")[-1]]
    else:
        raise RuntimeError(
            "Please ensure model path has mapping in `config.py`!"
            f"\n\tModel Path: `{model_path_or_name}`")

    # Ensure model name is valid, if online model is chosen
    if is_provider_online(model_provider):
        assert model_name in MODEL_INFO['online_model'], (
            f"Online model provided `{model_name}` is invalid! "
            f"\nValid options: {MODEL_INFO['online_model']}"
        )

    return model_name, model_path


################################################################################
#                              Plotting Functions                              #
################################################################################
def get_dataset_directory(dataset_name):
    """
    Retrieve the directory path for a given dataset name.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset

    Returns
    -------
    str
        The directory path corresponding to the dataset name.
    """
    for dataset_names, dir_data in config.DATASET_TO_DIR.items():
        if dataset_name in dataset_names:
            return dir_data
    raise RuntimeError(f"Failed to find dataset directory for `{dataset_name}`!")


def get_expected_dataset_files(dataset_name):
    """
    Retrieve the expected dataset files for a given dataset name.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset

    Returns
    -------
    list
        A list of expected dataset files for the given dataset name.
    """
    dir_data = get_dataset_directory(dataset_name)
    dir_dataset = os.path.join(dir_data, dataset_name)
    fnames = [
        fname for fname in os.listdir(dir_dataset)
        if os.path.isfile(os.path.join(dir_dataset, fname))
    ]
    return fnames


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    # Configure multiprocessing; to avoid vLLM issues with multi-threading
    multiprocessing.set_start_method('spawn')
    Fire({
        "generate": generate,
        "bias_evaluate": bias_eval_dataset_collection,
        "find_unfinished": find_unfinished,
        "delete": delete,
    })
