"""
quantize_models.py

Description: Used to quantize LLMs with AWQ and GPTQ.
"""

# Standard libraries
import json
import os
import logging
import random
import sys

# Non-standard libraries
import numpy as np
import torch
from awq import AutoAWQForCausalLM
from datasets import load_dataset
from fire import Fire
from huggingface_hub import HfApi
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig, TextStreamer


################################################################################
#                                    Config                                    #
################################################################################
# Max GPU memory (for 1 GPU)
MAX_MEMORY = None       # {0: "77GB"}

# Flag to overwrite existing HF models
OVERWRITE = False


################################################################################
#                                  Constants                                   #
################################################################################
# Create logger
LOGGER = logging.getLogger(__name__)
# Set logger level
LOGGER.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

# Get device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model location to new name
MODELS_RENAMED = {
    "meta-llama/Meta-Llama-3.1-8B": "Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct",

    "meta-llama/Llama-3.1-70B-Instruct": "Meta-Llama-3.1-70B-Instruct",

    "meta-llama/Llama-3.2-3B": "Meta-Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Meta-Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision": "Meta-Llama-3.2-11B-Vision",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "Meta-Llama-3.2-11B-Vision-Instruct",

    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3",
    "google/gemma-7b-it": "Gemma-7B-Instruct",
    "microsoft/Phi-3-small-8k-instruct": "Phi-3-small-7B-Instruct",
}

# HuggingFace API
HF_API = HfApi()

# HuggingFace User ID
HF_USER_ID = os.environ["HF_DATA_USERNAME"]


################################################################################
#                             Quantization Configs                             #
################################################################################
# Default AWQ Config
DEFAULT_AWQ_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "version": "GEMM",
}

# Default GPTQ Config
DEFAULT_GPTQ_CONFIG = {
    "bits": 4,  # quantize model to 4-bit
    "dataset": "wikitext2",
    "model_seqlen": 2048,
    "group_size": 128,  # it is recommended to set the value to 128
    "desc_act": True,  # set to False can significantly speed up inference but the perplexity may slightly bad
    "sym": True,  # using symmetric quantization so that the range is symmetric allowing the value 0 to be precisely represented (can provide speedups)
    "damp_percent": 0.1,  # see https://github.com/AutoGPTQ/AutoGPTQ/issues/196,
    "cache_block_outputs": False,        # TODO: Set to False if using GPU
}


################################################################################
#                               Set random seeds                               #
################################################################################
random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)


################################################################################
#                               Helper Functions                               #
################################################################################
def create_hub_save_path(model_path, q_method, q_bits):
    """
    Create save path for quantized model in HuggingFace Hub.

    Parameters
    ----------
    model_path : str
        Name of model in HuggingFace/locally
    q_method : str
        Quantization method to use
        (one of ["awq", "gptq", "lc-gpt"])
    q_bits : int
        Number of bits to quantize model to

    Returns
    -------
    str
        Name of quantized model directory
    """
    # Create hub save path
    # NOTE: If renamed explicitly, simply use that
    if model_path in MODELS_RENAMED:
        model_name = MODELS_RENAMED[model_path]
    else:
        model_name = model_path.split("/")[-1]
    hub_save_path = f"{model_name}-{q_method.upper()}-{q_bits}bit"
    return hub_save_path


################################################################################
#                                Main Functions                                #
################################################################################
def awq(model_path, save_to_hub=False, **quant_config):
    """
    Quantize model with AWQ.

    Parameters
    ----------
    model_path : str
        Name of model in HuggingFace/locally
        (e.g., meta-llama/Meta-Llama-3-8B-Instruct)
    save_to_hub : bool
        If True, save quantized model to HuggingFace Hub, by default False
    quant_config : **kwargs
        Quantization configuration parameters
        (e.g., zero_point, q_group_size, w_bit, version)
    """
    LOGGER.info("""
################################################################################
#                               AWQ Quantization                               #
################################################################################
Quantizing model `%s`
""", model_path)

    # Add default arguments
    for k, v in DEFAULT_AWQ_CONFIG.items():
        if k not in quant_config:
            quant_config[k] = v

    # Create path to save to in HuggingFace
    hub_save_path = create_hub_save_path(model_path, "awq", quant_config["w_bit"])

    # Name of repository to save to: "USER-ID / MODEL-PATH"
    repo_id = f"{HF_USER_ID}/{hub_save_path}"

    # Skip, if already done
    if not OVERWRITE and HF_API.repo_exists(repo_id):
        LOGGER.info("Model `%s` already quantized", hub_save_path)
        return

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )

    # Quantize
    # NOTE: If calibration data not specified, uses PILE validation set
    model.quantize(tokenizer, quant_config=quant_config)

    # Create configuration to make it compatible with transformers
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    # Save configuration to model
    model.model.config.quantization_config = quantization_config

    # Save quantized model locally
    model.save_quantized(repo_id)
    tokenizer.save_pretrained(repo_id)

    # Save quantization configuartion locally
    with open(f"{repo_id}/quant_config.json", "w") as f:
        json.dump(quant_config, f, indent=4)

    # Push to HuggingFace Hub
    if save_to_hub:
        LOGGER.info("Pushing quantized model to HuggingFace Hub `%s`...", hub_save_path)
        commit_message = f"AWQ model for {model_path}: {quant_config}"

        #  1. Create folder
        HF_API.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )
        # 2. Upload quantized model
        HF_API.upload_folder(
            repo_id=repo_id,
            folder_path=repo_id,
            commit_message=commit_message,
        )

        # 3. Save tokenizer to hub
        tokenizer.save_pretrained(repo_id)

        # NOTE: Following are not implemented in AutoAWQ
        model.push_to_hub(hub_save_path, commit_message=commit_message)
        tokenizer.push_to_hub(hub_save_path)


def gptq(model_path, save_to_hub=False, **quant_config):
    """
    Quantize model with GPTQ.

    Parameters
    ----------
    model_path : str
        Name of model in HuggingFace/locally
        (e.g., meta-llama/Meta-Llama-3-8B-Instruct)
    save_to_hub : bool
        If True, save quantized model to HuggingFace Hub, by default False
    quant_config : **kwargs
        Quantization configuration parameters
        (e.g., bits, group_size, desc_act)
    """
    LOGGER.info("""
################################################################################
#                              GPTQ Quantization                               #
################################################################################
Quantizing model `%s`
""", model_path)
    # Add default arguments
    for k, v in DEFAULT_GPTQ_CONFIG.items():
        if k not in quant_config:
            quant_config[k] = v
    LOGGER.info(quant_config)

    # Create path to save to in HuggingFace
    hub_save_path = create_hub_save_path(model_path, "gptq", quant_config["bits"])

    # Name of repository to save to: "USER-ID / MODEL-PATH"
    repo_id = f"{HF_USER_ID}/{hub_save_path}"

    # Skip, if already done
    if not OVERWRITE and HF_API.repo_exists(repo_id):
        LOGGER.info("Model `%s` already quantized", hub_save_path)
        return

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto", # TODO: Check if this should be torch.float16,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Perform GPTQ quantization
    curr_quant_config = DEFAULT_GPTQ_CONFIG.copy()
    curr_quant_config.update(quant_config)
    quantizer = GPTQQuantizer(**curr_quant_config)
    q_model = quantizer.quantize_model(model, tokenizer)

    # Save to HF Hub
    if save_to_hub:
        LOGGER.info("Pushing quantized model to HuggingFace Hub `%s`...", hub_save_path)
        commit_message = f"GPTQ model for {model_path}: {quant_config}"
        q_model.push_to_hub(
            hub_save_path,
            commit_message=commit_message,
            use_safetensors=True,
        )
        tokenizer.push_to_hub(hub_save_path)


def smooth_gptq(
        model_path,
        num_gpus=1,
        scheme="W4A16",
        num_samples=512,
        max_seq_len=6144,
        damp_factor=0.01,
        actorder=True,
        smoothing_strength=0.8,
        save_to_hub=False,
    ):
    """
    Quantize model with SmoothQuant and GPTQ.

    Parameters
    ----------
    model_path : str
        Name of model in HuggingFace/locally
    num_gpus : int
        Number of GPUs
    scheme : str
        GPTQ quantization scheme (W4A16, W8A8, W8A16)
    num_samples : int
        Number of calibration samples
    max_seq_len : int
        Maximum sequence length
    damp_factor : float
        Dampening factor
    smoothing_strength : float
        Smoothing strength
    save_to_hub : bool, optional
        If True, save quantized model to HuggingFace Hub, by default False
    """
    LOGGER.info("""
################################################################################
#                        SmoothQuant-GPTQ Quantization                         #
################################################################################
Quantizing model `%s`
""", model_path)

    # Create hub save path
    # NOTE: If renamed explicitly, simply use that
    if model_path in MODELS_RENAMED:
        model_name = MODELS_RENAMED[model_path]
    else:
        model_name = model_path.split("/")[-1]
    hub_save_path = f"{model_name}-LC-SmoothQuant-GPTQ-{scheme}"

    # Name of repository to save to: "USER-ID / MODEL-PATH"
    repo_id = f"{HF_USER_ID}/{hub_save_path}"

    # Skip, if already done
    if not OVERWRITE and HF_API.repo_exists(repo_id):
        LOGGER.info("Model `%s` already quantized", hub_save_path)
        return

    # Prepare dataset
    preprocess_fn = lambda example: {"text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{text}".format_map(example)}
    dataset_name = "neuralmagic/LLM_compression_calibration"
    dataset = load_dataset(dataset_name, split="train")
    ds = dataset.shuffle().select(range(num_samples))
    ds = ds.map(preprocess_fn)

    # TODO: Understand the impact of using a chat vs. non-chat calibration set

    # Prepare quantization recipe
    recipe = [
        SmoothQuantModifier(smoothing_strength=smoothing_strength),
        GPTQModifier(
            targets="Linear",
            scheme=scheme,
            ignore=["lm_head"],
            dampening_frac=damp_factor,
            actorder=actorder,
        )
    ]

    # Specify device mapping
    device_map = calculate_offload_device_map(
        model_path,
        num_gpus=num_gpus,
        reserve_for_hessians=True,
        torch_dtype="auto",
    )

    # Load model
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    # Perform quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_samples,
        output_dir=hub_save_path,
    )

    # Save model and tokenizer locally first
    model.save_pretrained(hub_save_path, push_to_hub=False, save_compressed=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.save_pretrained(hub_save_path, push_to_hub=False)

    # Now, push local folder to Hub
    if save_to_hub:
        #  1. Create folder
        HF_API.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )
        # 2. Upload quantized model
        HF_API.upload_folder(
            repo_id=repo_id,
            folder_path=hub_save_path,
            commit_message="Push folder to HuggingFace Hub",
        )


def smooth_rtn(
        model_path,
        smoothquant=False,
        scheme="W4A16",
        kv_scheme="KV16",
        smoothing_strength=0.8,
        num_samples=512,
        max_seq_len=6144,
        save_to_hub=False,
    ):
    """
    Quantize model with Round-to-Nearest and if specified, SmoothQuant.

    Note
    ----
    See the following page for all quantization schemes.
    https://github.com/neuralmagic/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py

    Parameters
    ----------
    model_path : str
        Name of model in HuggingFace/locally
    smoothquant : bool
        If True, use SmoothQuant
    scheme : str
        Quantization scheme (W4A16, W8A8, W8A16, FP8)
    kv_scheme : str
        Key-value cache quantization scheme (e.g., KV4, KV8, KV16, KV32)
    smoothing_strength : float
        Smoothing strength
    num_samples : int
        For SmoothQuant, number of calibration samples
    max_seq_len : int
        For SmoothQuant, maximum sequence length
    save_to_hub : bool, optional
        If True, save quantized model to HuggingFace Hub, by default False
    """
    LOGGER.info("""
################################################################################
#                         SmoothQuant-RTN Quantization                         #
################################################################################
Quantizing model `%s`
""", model_path)

    # Create hub save path
    # NOTE: If renamed explicitly, simply use that
    if model_path in MODELS_RENAMED:
        model_name = MODELS_RENAMED[model_path]
    else:
        model_name = model_path.split("/")[-1]

    # Create hub save path
    if smoothquant:
        hub_save_path = f"{model_name}-LC-SmoothQuant-RTN-{scheme}"
    else:
        hub_save_path = f"{model_name}-LC-RTN-{scheme}"

    # Append KV scheme, if different from KV16
    quantizing_kv = (kv_scheme != "KV16")
    if quantizing_kv:
        hub_save_path = f"{hub_save_path}-{kv_scheme}"

    # Name of repository to save to: "USER-ID / MODEL-PATH"
    repo_id = f"{HF_USER_ID}/{hub_save_path}"

    # Skip, if already done
    if not OVERWRITE and HF_API.repo_exists(repo_id):
        LOGGER.info("Model `%s` already quantized", hub_save_path)
        return

    # Prepare quantization recipe
    recipe = []
    # 1. Add SmoothQuant, if specified
    if smoothquant:
        recipe.append(SmoothQuantModifier(smoothing_strength=smoothing_strength))
    # 2. Setup Quantization Modifier (w/ KV cache quantization, if specified)
    quant_modifier_kwargs = {}
    if quantizing_kv:
        quant_modifier_kwargs["kv_cache_scheme"] = {
            "num_bits": int(kv_scheme.replace("KV", "")),
            "type": "float",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        }
    recipe.append(QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=["lm_head"],
        **quant_modifier_kwargs,
    ))

    # Load model
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Prepare dataset if SmoothQuant or quantizing KV cache
    kwargs = {}
    if smoothquant or quantizing_kv:
        # Prepare dataset
        preprocess_fn = lambda example: {"text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{text}".format_map(example)}
        dataset_name = "neuralmagic/LLM_compression_calibration"
        dataset = load_dataset(dataset_name, split="train")
        ds = dataset.shuffle().select(range(num_samples))
        ds = ds.map(preprocess_fn)
        kwargs["dataset"] = ds
        kwargs["max_seq_length"] = max_seq_len
        kwargs["num_calibration_samples"] = num_samples

    # Perform quantization
    oneshot(
        model=model,
        recipe=recipe,
        tokenizer=tokenizer,
        output_dir=hub_save_path,
        **kwargs,
    )

    # Save model and tokenizer locally first
    model.save_pretrained(hub_save_path, push_to_hub=False, save_compressed=True)
    tokenizer.save_pretrained(hub_save_path, push_to_hub=False)

    # Now, push local folder to Hub
    if save_to_hub:
        #  1. Create folder
        HF_API.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )
        # 2. Upload quantized model
        HF_API.upload_folder(
            repo_id=repo_id,
            folder_path=hub_save_path,
            commit_message="Push folder to HuggingFace Hub",
        )


def main(model_path, q_method, scheme, num_gpus=1):
    """
    Main entry point for quantizing a model.

    Parameters
    ----------
    model_path : str
        Name of model in HuggingFace/locally
    q_method : str
        Quantization method to use
        (one of ["awq", "gptq", "smooth_gptq"])
    scheme : str
        Quantization scheme (W4A16, W8A8, W8A16)
    num_gpus : int
        Number of GPUs to use
    """
    q_method_to_func = {
        "awq": awq,
        "gptq": gptq,
        "smooth_gptq": smooth_gptq,
        "rtn": smooth_rtn,
    }
    assert q_method in q_method_to_func, f"Invalid quantization method: {q_method}"

    # Get quantization function
    func =  q_method_to_func[q_method]

    # Quantize model
    # CASE 1: AWQ
    if q_method == "awq":
        assert scheme == "W4A16"
        func(model_path, w_bit=scheme)
    # CASE 2: GPTQ
    elif q_method == "gptq":
        assert scheme.endswith("A16")
        bits = int(bits.split("W")[-1].split("A16")[0])
        func(model_path, bits=bits) 
    # CASE 3: Smooth GPTQ
    elif q_method == "smooth_gptq":
        func(model_path, scheme=scheme, num_gpus=num_gpus)
    # CASE 4: RTN
    elif q_method == "rtn":
        func(model_path, smoothquant=True, scheme=scheme)
    func(model_path, bits=scheme)


if __name__ == "__main__":
    Fire({
        "awq": awq,
        "gptq": gptq,
        "smooth_gptq": smooth_gptq,
        "rtn": smooth_rtn,
    })
