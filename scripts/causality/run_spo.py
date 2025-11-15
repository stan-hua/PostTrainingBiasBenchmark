"""
run_spo.py

Description: Fine-tune model using Simple Preference Optimization

Usage:

# Reduce uncertainty (standard gradient descent)
python train_simpo.py train \
    --gradient_ascent=False \
    --merge

# Increase uncertainty (gradient ascent)
python train_simpo.py train \
    --gradient_ascent=True \
    --merge
"""

# Standard libraries
import json
import logging
import os
from typing import Tuple, Dict, Any, Optional

# Non-standard libraries
import fire
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer

# Custom libraries
import config


################################################################################
#                                  Constants                                   #
################################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default training dataset
TRAIN_PATH = config.CAUSALITY_PATHS["train_set"]
OUTPUT_DIR = config.CAUSALITY_PATHS["output_dir"]

# Use Comet for logging
USE_COMET_ML = True


################################################################################
#                                   Classes                                    #
################################################################################
class CPOGradientAscentTrainer(CPOTrainer):
    """
    Modified SimPO Trainer that supports gradient ascent.

    Parameters
    ----------
    gradient_ascent : bool
        If True, perform gradient ascent (maximize loss) instead of descent.
        This makes the model equally uncertain between chosen and rejected responses.

    Notes
    -----
    Gradient ascent is useful for increasing model uncertainty, making two
    response options equally likely. This is achieved by negating the loss,
    which causes the optimizer to maximize rather than minimize.
    """

    def __init__(self, *args, gradient_ascent=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_ascent = gradient_ascent

        if self.gradient_ascent:
            logger.info("=" * 60)
            logger.info("GRADIENT ASCENT MODE ENABLED")
            logger.info("Training will MAXIMIZE loss (increase uncertainty)")
            logger.info("=" * 60)

    def cpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute CPO/SimPO loss with optional gradient ascent.
        
        Parameters
        ----------
        policy_chosen_logps : torch.FloatTensor
            Log probabilities of chosen responses under policy model.
            Shape: (batch_size,)
        policy_rejected_logps : torch.FloatTensor
            Log probabilities of rejected responses under policy model.
            Shape: (batch_size,)
        
        Returns
        -------
        tuple of (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor)
            Tuple containing (losses, chosen_rewards, rejected_rewards).
            Losses are negated if gradient_ascent=True.
        """
        # Compute standard CPO/SimPO loss using parent method
        losses, chosen_rewards, rejected_rewards = super().cpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )
        
        # Negate loss for gradient ascent
        if self.gradient_ascent:
            losses = -losses
        
        return losses, chosen_rewards, rejected_rewards


################################################################################
#                                  Functions                                   #
################################################################################
def get_default_config():
    """
    Get default training configuration.

    Returns
    -------
    dict
        Dictionary containing all default training parameters.
    """
    return {
        # Model
        'model_name': 'Qwen/Qwen2.5-0.5B-Instruct',

        # Data
        'train_data_path': TRAIN_PATH,
        'output_dir': OUTPUT_DIR,

        # LoRA config
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'lora_target_modules': [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ],

        # Training
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 8,
        'learning_rate': 5e-5,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,

        # SimPO-specific (via CPO)
        'loss_type': 'simpo',
        'beta': 2.0,
        'simpo_gamma': 1.0,
        'cpo_alpha': 0,
        'gradient_ascent': False,

        # Hardware
        'fp16': True,
        'dataloader_num_workers': 4,

        # Logging & Saving
        'seed': 42,
        'logging_steps': 10,
        'save_steps': 100,
        'save_total_limit': 3,
        'report_to': "none",

        # Comet-specific (optional)
        'comet_project_name': None,
        'comet_experiment_name': None,
        'comet_tags': None,
    }


def format_dataset_for_simpo(data_path, tokenizer):
    """
    Convert preference pairs to SimPO format.

    Parameters
    ----------
    data_path : str
        Path to CSV file containing preference pairs.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for applying chat template.

    Returns
    -------
    datasets.Dataset
        Formatted dataset ready for SimPO training.

    Notes
    -----
    Expected input format:
        {
            "prompt": "question text",
            "accept_response": "preferred response",
            "reject_response": "dispreferred response",
            ...
        }
    """
    # Load dataset and convert to list of dict
    df = pd.read_csv(data_path)
    data = df.to_dict(orient='records')

    formatted_data = []
    for item in data:
        # Format as chat messages
        prompt_messages = [
            {"role": "user", "content": item['prompt']}
        ]

        accepted_messages = prompt_messages + [
            {"role": "assistant", "content": item['accept_response']}
        ]

        rejected_messages = prompt_messages + [
            {"role": "assistant", "content": item['reject_response']}
        ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        chosen = tokenizer.apply_chat_template(
            accepted_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        rejected = tokenizer.apply_chat_template(
            rejected_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        formatted_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return Dataset.from_list(formatted_data)


def get_lora_model(model_name, lora_config_dict):
    """
    Load model and apply LoRA adapters.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    lora_config_dict : dict
        Dictionary containing LoRA configuration parameters.

    Returns
    -------
    peft.PeftModel
        Model with LoRA adapters applied.
    """
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Create LoRA config from dict
    lora_config = LoraConfig(
        r=lora_config_dict['lora_r'],
        lora_alpha=lora_config_dict['lora_alpha'],
        target_modules=lora_config_dict['lora_target_modules'],
        lora_dropout=lora_config_dict['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} "
        f"({100*trainable_params/total_params:.2f}%)"
    )

    return model


def setup_comet_logging(config):
    """
    Setup Comet ML logging with custom configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing Comet settings.

    Returns
    -------
    dict
        Updated config with Comet environment variables set.
    """
    if config['report_to'] == 'comet_ml' or config['report_to'] == 'comet':
        config['report_to'] = 'comet_ml'

        if config.get('comet_project_name'):
            os.environ['COMET_PROJECT_NAME'] = config['comet_project_name']

        if config.get('comet_experiment_name'):
            os.environ['COMET_EXPERIMENT_NAME'] = config['comet_experiment_name']

        os.environ['COMET_LOG_ASSETS'] = 'False'

        logger.info("=" * 60)
        logger.info("Comet ML Configuration:")
        logger.info(f"  Project: {config.get('comet_project_name', 'default')}")
        logger.info(f"  Experiment: {config.get('comet_experiment_name', 'auto-generated')}")
        logger.info("=" * 60)

    return config


def create_output_path(base_output_dir, model_name, gradient_ascent):
    """
    Create output path with model name and gradient ascent/descent suffix.
    
    Parameters
    ----------
    base_output_dir : str
        Base output directory.
    model_name : str
        HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-0.5B-Instruct').
    gradient_ascent : bool
        Whether gradient ascent is being used.
    
    Returns
    -------
    str
        Full output path with model name and suffix.
    
    Examples
    --------
    >>> create_output_path('outputs', 'Qwen/Qwen2.5-0.5B-Instruct', False)
    'outputs/qwen2.5-0.5b-instruct_gd'
    
    >>> create_output_path('outputs', 'Qwen/Qwen2.5-0.5B-Instruct', True)
    'outputs/qwen2.5-0.5b-instruct_ga'
    """
    # Extract model name (last part after /)
    model_short_name = model_name.split('/')[-1].lower()
    
    # Create suffix based on gradient ascent/descent
    suffix = 'ga' if gradient_ascent else 'gd'
    
    # Combine: base_dir/model_name_suffix
    output_path = os.path.join(base_output_dir, f"{model_short_name}_{suffix}")
    
    return output_path


################################################################################
#                                Main Functions                                #
################################################################################
def train(config):
    """
    Main training function for SimPO with LoRA.

    Parameters
    ----------
    config : dict
        Training configuration dictionary.

    Returns
    -------
    tuple of (peft.PeftModel, transformers.PreTrainedTokenizer)
        Trained model and tokenizer.

    Notes
    -----
    If config['gradient_ascent'] is True, training will maximize loss instead
    of minimizing it, which increases model uncertainty between responses.

    SimPO is implemented via CPOTrainer with loss_type="simpo".
    """
    # Setup Comet logging if enabled
    config = setup_comet_logging(config)

    # Set seed
    torch.manual_seed(config['seed'])

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    logger.info(f"Loading training data from {config['train_data_path']}")
    train_dataset = format_dataset_for_simpo(config['train_data_path'], tokenizer)
    logger.info(f"Training samples: {len(train_dataset)}")

    # Load model with LoRA
    model = get_lora_model(config['model_name'], config)

    # Configure CPO training with SimPO loss
    training_args = CPOConfig(
        output_dir=config['output_dir'],

        # Training
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],

        # Optimization
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config['lr_scheduler_type'],
        warmup_ratio=config['warmup_ratio'],
        optim="adamw_torch",
        weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'],

        # SimPO-specific (via CPO)
        loss_type=config['loss_type'],  # "simpo"
        beta=config['beta'],
        simpo_gamma=config['simpo_gamma'],
        cpo_alpha=config['cpo_alpha'],  # Set to 0 for pure SimPO

        # Hardware
        fp16=config['fp16'],
        dataloader_num_workers=config['dataloader_num_workers'],

        # Logging & Saving
        logging_steps=config['logging_steps'],
        save_strategy="steps",
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],

        # Misc
        seed=config['seed'],
        remove_unused_columns=False,
        report_to=config['report_to'],
    )

    # Initialize trainer (with gradient ascent support)
    trainer = CPOGradientAscentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        gradient_ascent=config['gradient_ascent'],
    )

    # Log hyperparameters to Comet
    if config['report_to'] == 'comet_ml':
        try:
            experiment = trainer.state.comet_experiment

            experiment.log_parameters({
                'lora_r': config['lora_r'],
                'lora_alpha': config['lora_alpha'],
                'loss_type': config['loss_type'],
                'beta': config['beta'],
                'simpo_gamma': config['simpo_gamma'],
                'cpo_alpha': config['cpo_alpha'],
                'gradient_ascent': config['gradient_ascent'],
                'effective_batch_size': (
                    config['per_device_train_batch_size'] *
                    config['gradient_accumulation_steps']
                ),
            })

            if config.get('comet_tags'):
                experiment.add_tags(config['comet_tags'])

            logger.info("Custom hyperparameters logged to Comet")
        except Exception as e:
            logger.warning(f"Could not log to Comet: {e}")

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {config['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config['output_dir'])

    logger.info("Training complete!")

    return model, tokenizer


def merge_and_save(lora_model_path, output_path, base_model_name='Qwen/Qwen2.5-0.5B-Instruct'):
    """
    Merge LoRA weights into base model and save.

    Parameters
    ----------
    lora_model_path : str
        Path to directory containing LoRA adapters.
    output_path : str
        Path where merged model will be saved.
    base_model_name : str, optional
        HuggingFace identifier for base model.

    Returns
    -------
    transformers.PreTrainedModel
        Merged model with LoRA weights integrated.
    """
    logger.info(f"Merging LoRA weights from {lora_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, lora_model_path)

    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete!")

    return merged_model


def run_training(
    train_data=None,
    output_dir=None,
    model_name='Qwen/Qwen2.5-0.5B-Instruct',
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    epochs=3,
    batch_size=4,
    grad_accum=8,
    lr=5e-5,
    beta=2.0,
    simpo_gamma=1.0,
    gradient_ascent=False,
    seed=42,
    merge=False,
    config_file=None,
    comet=USE_COMET_ML,
    comet_project=None,
    comet_experiment=None,
    comet_tags=None,
):
    """
    Run SimPO training with LoRA fine-tuning.

    Parameters
    ----------
    train_data : str
        Path to training data CSV file.
    output_dir : str
        Output directory for trained model.
    model_name : str, optional
        HuggingFace model identifier.
    lora_r : int, optional
        LoRA rank (default: 16).
    lora_alpha : int, optional
        LoRA alpha scaling factor (default: 32).
    lora_dropout : float, optional
        LoRA dropout rate (default: 0.05).
    epochs : int, optional
        Number of training epochs (default: 3).
    batch_size : int, optional
        Per-device training batch size (default: 4).
    grad_accum : int, optional
        Gradient accumulation steps (default: 8).
    lr : float, optional
        Learning rate (default: 5e-5).
    beta : float, optional
        SimPO beta temperature parameter (default: 2.0).
    simpo_gamma : float, optional
        SimPO gamma parameter for target reward margin (default: 1.0).
    gradient_ascent : bool, optional
        Enable gradient ascent to maximize uncertainty (default: False).
        When True, training maximizes loss, making chosen and rejected
        responses equally likely.
    seed : int, optional
        Random seed (default: 42).
    merge : bool, optional
        Whether to merge LoRA weights after training (default: False).
    config_file : str, optional
        Path to JSON config file (default: None).
    comet : bool, optional
        Enable Comet ML logging (default: False).
    comet_project : str, optional
        Comet project name (default: None).
    comet_experiment : str, optional
        Comet experiment name (default: None).
    comet_tags : str, optional
        Comma-separated tags for Comet (default: None).

    Returns
    -------
    dict
        Dictionary containing paths to saved models.

    Examples
    --------
    >>> # Standard training (reduce uncertainty)
    >>> run_training(
    ...     train_data='data/train.json',
    ...     output_dir='outputs/reduce_uncertainty',
    ...     gradient_ascent=False
    ... )

    >>> # Gradient ascent (increase uncertainty)
    >>> run_training(
    ...     train_data='data/train.json',
    ...     output_dir='outputs/increase_uncertainty',
    ...     gradient_ascent=True
    ... )

    >>> # With Comet logging
    >>> run_training(
    ...     train_data='data/train.json',
    ...     output_dir='outputs/model1',
    ...     gradient_ascent=True,
    ...     comet=True,
    ...     comet_project='uncertainty-quantization',
    ...     comet_experiment='gradient_ascent_experiment',
    ...     comet_tags='gradient-ascent,increase-uncertainty'
    ... )
    """
    # Start with default config
    config = get_default_config()

    # Load config file if provided
    if config_file is not None:
        logger.info(f"Loading config from {config_file}")
        with open(config_file, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)

    # Create output path with model name and gradient suffix
    config["train_data_path"] = train_data or TRAIN_PATH
    config["output_dir"] = output_dir or OUTPUT_DIR
    config["output_dir"] = create_output_path(config["output_dir"], model_name, gradient_ascent)

    # Override with command line arguments
    config.update({
        'model_name': model_name,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'num_train_epochs': epochs,
        'per_device_train_batch_size': batch_size,
        'gradient_accumulation_steps': grad_accum,
        'learning_rate': lr,
        'beta': beta,
        'simpo_gamma': simpo_gamma,
        'gradient_ascent': gradient_ascent,
        'seed': seed,
    })

    # Setup Comet logging
    if comet:
        config['report_to'] = 'comet_ml'
        config['comet_project_name'] = comet_project
        config['comet_experiment_name'] = comet_experiment
        if comet_tags:
            config['comet_tags'] = comet_tags.split(',')

    # Log configuration
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info("=" * 60)
    for key, value in config.items():
        if not isinstance(value, list):
            logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    # Train
    model, tokenizer = train(config)

    result = {
        'lora_path': config['output_dir'],
    }

    # Optionally merge
    if merge:
        merged_path = f"{config['output_dir']}_merged"
        logger.info(f"\nMerging LoRA weights to {merged_path}")
        merge_and_save(config['output_dir'], merged_path, config['model_name'])
        result['merged_path'] = merged_path

    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"LoRA adapters saved to: {result['lora_path']}")
    if merge:
        logger.info(f"Merged model saved to: {result['merged_path']}")
    if comet:
        logger.info(f"View results at: https://www.comet.com/{comet_project}")
    logger.info("=" * 60)

    return result


def merge_existing_lora(lora_path, output_path, base_model='Qwen/Qwen2.5-0.5B-Instruct'):
    """
    Merge existing LoRA weights.

    Parameters
    ----------
    lora_path : str
        Path to directory containing LoRA adapters.
    output_path : str
        Path where merged model will be saved.
    base_model : str, optional
        HuggingFace identifier for base model.

    Returns
    -------
    str
        Path to merged model.
    """
    merge_and_save(lora_path, output_path, base_model)
    return output_path


if __name__ == "__main__":
    fire.Fire({
        'train': run_training,
        'merge': merge_existing_lora,
    })
