"""
extract_quant_errors.py

Description: Used to compute quantization errors between original and
             quantized models.
"""

# Standard libraries
import json
import gc
import psutil
import os
import warnings
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, Tuple, List, Optional

# Non-standard libraries
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Custom libraries
import config


################################################################################
#                                  Constants                                   #
################################################################################
warnings.filterwarnings("ignore")

# HF name where models are stored
HF_DATA_USERNAME = os.environ["HF_DATA_USERNAME"]
assert HF_DATA_USERNAME, ("Please set HF_DATA_USERNAME environment variable "
    "with HuggingFace username, where models are stored")

# Mapping of model path to name
MODEL_PATH_TO_NAME = config.MODEL_INFO["model_path_to_name"]

# Number of processes
NUM_PROCESSES = 2

# Mapping of unquantized model to quantized variants
BASE_TO_QUANTIZED = {
    'meta-llama/Llama-3.1-8B-Instruct': [
        'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
        'neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16',
        f'{HF_DATA_USERNAME}/Llama-3.1-8B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Llama-3.1-8B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Llama-3.1-8B-Instruct-LC-SmoothQuant-RTN-W4A16',
    ],
    'meta-llama/Llama-3.2-1B-Instruct': [
        f'{HF_DATA_USERNAME}/Llama-3.2-1B-Instruct-AWQ-W4A16',
        f'{HF_DATA_USERNAME}/Llama-3.2-1B-Instruct-LC-GPTQ-W4A16',
        f'{HF_DATA_USERNAME}/Llama-3.2-1B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Llama-3.2-1B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Llama-3.2-1B-Instruct-LC-SmoothQuant-RTN-W4A16',
    ],
    'meta-llama/Llama-3.2-3B-Instruct': [
        f'{HF_DATA_USERNAME}/Meta-Llama-3.2-3B-Instruct-AWQ-W4A16',
        f'{HF_DATA_USERNAME}/Meta-Llama-3.2-3B-Instruct-LC-GPTQ-W4A16',
        f'{HF_DATA_USERNAME}/Meta-Llama-3.2-3B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Meta-Llama-3.2-3B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Meta-Llama-3.2-3B-Instruct-LC-SmoothQuant-RTN-W4A16',
    ],
    'mistralai/Ministral-8B-Instruct-2410': [
        f'{HF_DATA_USERNAME}/Ministral-8B-Instruct-2410-AWQ-W4A16',
        f'{HF_DATA_USERNAME}/Ministral-8B-Instruct-2410-LC-GPTQ-W4A16',
        f'{HF_DATA_USERNAME}/Ministral-8B-Instruct-2410-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Ministral-8B-Instruct-2410-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Ministral-8B-Instruct-2410-LC-SmoothQuant-RTN-W4A16',
    ],
    'Qwen/Qwen2-7B-Instruct': [
        'Qwen/Qwen2-7B-Instruct-AWQ',
        'Qwen/Qwen2-7B-Instruct-GPTQ-Int4',
        f'{HF_DATA_USERNAME}/Qwen2-7B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Qwen2-7B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Qwen2-7B-Instruct-LC-SmoothQuant-RTN-W4A16',
    ],
    'Qwen/Qwen2.5-0.5B-Instruct': [
        'Qwen/Qwen2.5-0.5B-Instruct-AWQ',
        'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',
        f'{HF_DATA_USERNAME}/Qwen2.5-0.5B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-0.5B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-0.5B-Instruct-LC-SmoothQuant-RTN-W4A16',
    ],
    'Qwen/Qwen2.5-1.5B-Instruct': [
        'Qwen/Qwen2.5-1.5B-Instruct-AWQ',
        'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4',
        f'{HF_DATA_USERNAME}/Qwen2.5-1.5B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-1.5B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-1.5B-Instruct-LC-SmoothQuant-RTN-W4A16',
    ],
    'Qwen/Qwen2.5-3B-Instruct': [
        'Qwen/Qwen2.5-3B-Instruct-AWQ',
        'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4',
        f'{HF_DATA_USERNAME}/Qwen2.5-3B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-3B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-3B-Instruct-LC-SmoothQuant-RTN-W4A16',
    ],
    'Qwen/Qwen2.5-7B-Instruct': [
        'Qwen/Qwen2.5-7B-Instruct-AWQ',
        'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
        f'{HF_DATA_USERNAME}/Qwen2.5-7B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-7B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-7B-Instruct-LC-SmoothQuant-RTN-W4A16',
    ],
    'Qwen/Qwen2.5-14B-Instruct': [
        'Qwen/Qwen2.5-14B-Instruct-AWQ',
        'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4',
        f'{HF_DATA_USERNAME}/Qwen2.5-14B-Instruct-LC-RTN-W4A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-14B-Instruct-LC-RTN-W8A16',
        f'{HF_DATA_USERNAME}/Qwen2.5-14B-Instruct-LC-SmoothQuant-RTN-W4A16'
    ],
}

# Save file for results
SAVE_DIR = os.path.join(config.DIR_SAVE_DATA, "accum_quant_errors")
os.makedirs(SAVE_DIR, exist_ok=True)


################################################################################
#                                    Class                                     #
################################################################################
class QuantizationErrorAnalyzer:
    def __init__(self, device='cpu', dtype=torch.float16):
        """
        Initialize the quantization error analyzer.

        Args:
            device: Device to use ('cpu' recommended for memory efficiency)
            dtype: Data type for computations (bfloat16 for precision)
        """
        self.device = device
        self.dtype = dtype
        self.original_model = None
        self.quantized_model = None

    def load_models(self, original_model_id: str, quantized_model_id: str):
        """
        Load both original and quantized models efficiently with automatic method detection.
        """
        print("Loading original model...")
        self.original_model = AutoModelForCausalLM.from_pretrained(
            original_model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        print("Loading quantized model...")
        self.quantized_model = AutoModelForCausalLM.from_pretrained(
            quantized_model_id,
            torch_dtype=None,  # Don't force dtype conversion
            device_map=self.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # Move models to eval mode
        self.original_model.eval()
        self.quantized_model.eval()

        print(f"Models loaded successfully on {self.device}")
        self._print_memory_usage()

        # Verify quantization is preserved
        self._verify_quantization_preserved()

    def _verify_quantization_preserved(self):
        """
        Verify that the quantized model actually has quantized weights.
        """
        print("\nVerifying quantization preservation...")

        quantized_layers_found = 0
        total_layers_checked = 0
        format_counts = defaultdict(int)

        for name, module in self.quantized_model.named_modules():
            if self._has_weights(module):
                total_layers_checked += 1

                try:
                    # Get weight info without extracting (faster)
                    _, debug_info = self.get_module_weights(self.quantized_model, name, debug=False)

                    if debug_info.get('is_quantized', False):
                        quantized_layers_found += 1
                        fmt = debug_info.get('quantization_format', 'unknown')
                        format_counts[fmt] += 1
                        if total_layers_checked <= 5:  # Show details for first few
                            print(f"  ✓ {name}: {fmt} format")
                    else:
                        if total_layers_checked <= 5:  # Show details for first few
                            fmt = debug_info.get('quantization_format', 'unknown')
                            print(f"  ✗ {name}: Not quantized ({fmt})")
                except Exception as e:
                    if total_layers_checked <= 5:
                        print(f"  ✗ {name}: Error checking quantization - {e}")
                    continue

                if total_layers_checked >= 20:  # Check first 20 layers only
                    break

        print(f"\nQuantization summary:")
        print(f"  Total layers checked: {total_layers_checked}")
        print(f"  Quantized layers found: {quantized_layers_found}")
        print(f"  Quantization formats detected:")
        for fmt, count in format_counts.items():
            print(f"    {fmt}: {count} layers")

        if quantized_layers_found == 0:
            print("WARNING: No quantized layers detected! The model may have been dequantized during loading.")
            print("This would explain why all errors are 0.")

        return quantized_layers_found > 0

    def _has_weights(self, module) -> bool:
        """Check if module has any form of weights."""
        weight_attrs = ['weight', 'qweight', 'weight_packed']
        return any(hasattr(module, attr) for attr in weight_attrs)

    def _print_memory_usage(self):
        """Print current memory usage."""
        if self.device == 'cpu':
            memory = psutil.virtual_memory()
            print(f"CPU Memory usage: {memory.percent:.1f}% ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
        else:
            print(f"GPU Memory usage: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        """Print current memory usage."""
        if self.device == 'cpu':
            memory = psutil.virtual_memory()
            print(f"CPU Memory usage: {memory.percent:.1f}% ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
        else:
            print(f"GPU Memory usage: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def get_module_weights(self, model, module_name: str, debug=False) -> Tuple[torch.Tensor, Dict]:
        """
        Extract weights from a specific module, handling quantized weights.
        Returns (weight_tensor, debug_info)
        """
        module = dict(model.named_modules())[module_name]
        debug_info = {
            'module_type': type(module).__name__,
            'has_weight': hasattr(module, 'weight'),
            'has_qweight': hasattr(module, 'qweight'),
            'has_scales': hasattr(module, 'scales'),
            'has_qzeros': hasattr(module, 'qzeros'),
            'has_dequantize': hasattr(module, 'dequantize'),
            'weight_dtype': None,
            'weight_shape': None,
            'is_quantized': False
        }

        weight = None

        # Handle standard weight attribute
        if hasattr(module, 'weight'):
            weight = module.weight
            debug_info['weight_dtype'] = str(weight.dtype)
            debug_info['weight_shape'] = tuple(weight.shape)

            # Check if it's a quantized tensor
            if hasattr(weight, 'dequantize'):
                if debug:
                    print(f"  Found quantized weight with dequantize method")
                weight = weight.dequantize().to(self.dtype)
                debug_info['is_quantized'] = True
            elif weight.dtype in [torch.int8, torch.int4, torch.uint8]:
                debug_info['is_quantized'] = True
                if debug:
                    print(f"  Found integer weight (quantized): {weight.dtype}")
                # For integer weights, we might need quantization parameters
                weight = weight.to(self.dtype)  # Simple cast (may not be accurate)
            else:
                weight = weight.to(self.dtype)

        # CASE: compressed-tensors
        elif hasattr(module, "compressor"):
            weight = module.compressor.decompress_module(module).to(self.dtype)

        if weight is not None:
            debug_info['final_dtype'] = str(weight.dtype)
            debug_info['final_shape'] = tuple(weight.shape)

        return weight, debug_info

    def compute_error_metrics(self, original_weight: torch.Tensor,
                            quantized_weight: torch.Tensor) -> Dict[str, float]:
        """
        Compute various error metrics between original and quantized weights.
        """
        # Ensure same shape
        if original_weight.shape != quantized_weight.shape:
            print(f"Warning: Shape mismatch - Original: {original_weight.shape}, Quantized: {quantized_weight.shape}")
            return {}

        # Move to same device and dtype for computation
        orig = original_weight.to(self.device, dtype=self.dtype).float()
        quant = quantized_weight.to(self.device, dtype=self.dtype).float()

        # Compute error
        error = orig - quant

        # Various metrics
        mse = torch.mean(error ** 2).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(error)).item()

        # Relative metrics (avoid division by zero)
        orig_norm = torch.norm(orig).item()
        if orig_norm > 1e-8:
            snr = 20 * np.log10(orig_norm / (torch.norm(error).item() + 1e-8))
        else:
            snr = None

        # Cosine similarity
        cos_sim = torch.cosine_similarity(
            orig.flatten().unsqueeze(0), 
            quant.flatten().unsqueeze(0), 
            dim=1
        ).item()

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'snr_db': float(snr),
            'cosine_similarity': float(cos_sim),
            'max_error': float(torch.max(torch.abs(error)).item()),
            'weight_magnitude': float(orig_norm),
        }

    def find_comparable_modules(self) -> List[Tuple[str, str]]:
        """
        Find modules that exist in both models and have weights.
        Returns list of (original_module_name, quantized_module_name) tuples.
        """
        orig_modules = dict(self.original_model.named_modules())
        quant_modules = dict(self.quantized_model.named_modules())

        comparable = []

        # Create mapping dictionaries for flexible matching
        orig_weight_modules = {}
        quant_weight_modules = {}

        # Collect modules with weights from original model
        for name, module in orig_modules.items():
            if hasattr(module, 'weight'):
                orig_weight_modules[name] = module

        # Collect modules with weights from quantized model
        for name, module in quant_modules.items():
            if self._has_weights(module):
                quant_weight_modules[name] = module

        print(f"Original model has {len(orig_weight_modules)} weight modules")
        print(f"Quantized model has {len(quant_weight_modules)} weight modules")

        # Try different matching strategies
        for orig_name in orig_weight_modules:
            quant_name = None
            # Strategy 1: Exact match
            if orig_name in quant_weight_modules:
                quant_name = orig_name
            # Strategy 2: Handle extra .model layer in quantized model
            elif f"model.{orig_name}" in quant_weight_modules:
                quant_name = f"model.{orig_name}"
            if quant_name:
                comparable.append((orig_name, quant_name))

        print(f"Found {len(comparable)} comparable module pairs")
        if len(comparable) > 0:
            print("Sample matches:")
            for i, (orig, quant) in enumerate(comparable[:5]):
                print(f"  {orig} -> {quant}")

        return comparable

    def analyze_all_modules(self, debug_first_few=3) -> Dict[str, Dict[str, float]]:
        """
        Analyze quantization error for all comparable modules.
        """
        if self.original_model is None or self.quantized_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        # First, find comparable modules
        comparable_modules = self.find_comparable_modules()
        print(f"Found {len(comparable_modules)} comparable module pairs")

        # Debug first few modules
        if debug_first_few > 0 and len(comparable_modules) > 0:
            print(f"\nDebugging first {debug_first_few} module pairs...")
            for i, (orig_name, quant_name) in enumerate(comparable_modules[:debug_first_few]):
                print(f"\n--- Debugging pair {i+1}: {orig_name} vs {quant_name} ---")
                self.debug_module_pair(orig_name, quant_name)

        results = {}
        processed_count = 0
        identical_count = 0

        print(f"\nAnalyzing all {len(comparable_modules)} module pairs...")

        for orig_name, quant_name in comparable_modules:
            try:
                # Extract weights
                orig_weight, _ = self.get_module_weights(self.original_model, orig_name)
                quant_weight, _ = self.get_module_weights(self.quantized_model, quant_name)

                if orig_weight is None or quant_weight is None:
                    continue

                # Compute metrics
                metrics = self.compute_error_metrics(orig_weight, quant_weight)
                if metrics:
                    # Use original module name as key for consistency
                    results[orig_name] = metrics
                    processed_count += 1

                    # Check if weights are identical
                    if metrics['mse'] < 1e-10:
                        identical_count += 1

                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count} modules... ({identical_count} appear identical)")
                        self._print_memory_usage()

                # Clean up to save memory
                del orig_weight, quant_weight
                if processed_count % 20 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Error processing module pair {orig_name} -> {quant_name}: {e}")
                continue

        print(f"Analysis complete. Processed {processed_count} modules.")
        print(f"WARNING: {identical_count} modules appear to have identical weights!")
        if identical_count > processed_count * 0.5:
            print("WARNING: More than half the modules are identical - this suggests an issue with dequantization!")

        return results

    def debug_module_pair(self, orig_name: str, quant_name: str):
        """
        Debug a pair of modules to understand what's happening with the weights.
        """
        print(f"ORIGINAL MODULE: {orig_name}")
        orig_weight, orig_debug = self.get_module_weights(self.original_model, orig_name, debug=True)
        for key, value in orig_debug.items():
            print(f"  {key}: {value}")

        print(f"\nQUANTIZED MODULE: {quant_name}")
        quant_weight, quant_debug = self.get_module_weights(self.quantized_model, quant_name, debug=True)
        for key, value in quant_debug.items():
            print(f"  {key}: {value}")

        if orig_weight is not None and quant_weight is not None:
            print(f"\nWEIGHT COMPARISON:")
            print(f"  Original shape: {orig_weight.shape}")
            print(f"  Quantized shape: {quant_weight.shape}")
            print(f"  Original dtype: {orig_weight.dtype}")
            print(f"  Quantized dtype: {quant_weight.dtype}")

            # Sample some values
            print(f"  Original sample values: {orig_weight.flatten()[:5].tolist()}")
            print(f"  Quantized sample values: {quant_weight.flatten()[:5].tolist()}")

            # Quick difference check
            if orig_weight.shape == quant_weight.shape:
                diff = torch.abs(orig_weight - quant_weight)
                print(f"  Max absolute difference: {torch.max(diff).item()}")
                print(f"  Mean absolute difference: {torch.mean(diff).item()}")
                print(f"  Are they identical? {torch.allclose(orig_weight, quant_weight, atol=1e-8)}")
            else:
                print("  Cannot compare - shape mismatch")
        else:
            print("  Could not extract weights for comparison")

    def save_results(self, results: Dict[str, Dict[str, float]], save_path: str):
        """
        Save results to file for later analysis.
        """
        with open(save_path, 'w') as handler:
            json.dump(results, handler, indent=4)


################################################################################
#                               Helper Functions                               #
################################################################################
def convert_path_to_name(model_path: str) -> str:
    """
    Convert a model path to its nickname using MODEL_PATH_TO_NAME mapping.

    Parameters
    ----------
    model_path : str
        Path to model
    """
    options = [model_path, model_path.split("/")[-1], f"{HF_DATA_USERNAME}/{model_path}"]
    for option in options:
        if option in MODEL_PATH_TO_NAME:
            return MODEL_PATH_TO_NAME[option]

    raise RuntimeError(
        f"Failed to identify model nickname for `{model_path}`!\n"
        "Please update in config.py / MODEL_INFO['path_to_name']"
    )


################################################################################
#                                  Main Flows                                  #
################################################################################
def analyze_model_pair(original_model_id, quantized_model_id):
    """
    Analyze a single pair of original and quantized models.

    Parameters
    ----------
    original_model_id : str
        HuggingFace model ID for the original model
    quantized_model_id : str
        HuggingFace model ID for the quantized model
    """
    # Initialize analyzer
    analyzer = QuantizationErrorAnalyzer(device='cpu', dtype=torch.float16)

    # Convert model paths to names
    orig_name = convert_path_to_name(original_model_id)
    quant_name = convert_path_to_name(quantized_model_id) 
    
    # Create save path
    save_dir = os.path.join(SAVE_DIR, orig_name, quant_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "quant_errors.json")

    # Skip, if already done
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"Results already exist at {save_path}, skipping...")
        return

    try:
        # Load models
        analyzer.load_models(original_model_id, quantized_model_id)
        # Analyze quantization error with debugging
        results = analyzer.analyze_all_modules(debug_first_few=3)
        # Save results
        analyzer.save_results(results, save_path)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        with open(os.path.join(save_dir, "error.log"), 'w') as handler:
            handler.write(str(e) + "\n\n")
            traceback.print_exc(file=handler)
        raise


def parallel_analyze(orig_models=None):
    """
    Analyze all pairs of unquantized and quantized models in parallel

    Parameters
    ----------
    orig_models : list of str
        List of original models to include
    """
    # Create pairs of all unquantized and quantized models
    model_pairs = []
    for orig_model, quant_models in BASE_TO_QUANTIZED.items():
        # If specified, only filter for specific original models
        if orig_models and orig_model not in orig_models:
            continue
        for quant_model in quant_models:
            model_pairs.append((orig_model, quant_model))

    # CASE 1: No multiprocessing
    if NUM_PROCESSES <= 1:
        for model_pair in model_pairs:
            analyze_model_pair(*model_pair)
        return

    # CASE 2: Multi-processing
    # Use multiprocessing to analyze in parallel
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.starmap(analyze_model_pair, model_pairs)


if __name__ == "__main__":
    from fire import Fire
    Fire()
