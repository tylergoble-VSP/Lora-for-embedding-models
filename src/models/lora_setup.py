"""
Used in: 04_LoRA_Configuration.ipynb, 05_Fine_Tuning_Training.ipynb
Purpose:
    Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    of the EmbeddingGemma model using PEFT.
"""

from peft import LoraConfig, get_peft_model, TaskType  # PEFT library for LoRA
from transformers import AutoModel  # Base model type
from typing import List, Optional  # Type hints


def freeze_base_model(model: AutoModel) -> AutoModel:
    """
    Freeze all parameters in the base model so they won't be updated during training.

    Args:
        model: The base EmbeddingGemma model.

    Returns:
        The same model with all parameters frozen (requires_grad=False).
    """

    # Set requires_grad=False for all parameters
    # This ensures only LoRA adapter weights will be trainable
    for param in model.parameters():
        param.requires_grad = False

    return model


def configure_lora(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> LoraConfig:
    """
    Create a LoRA configuration for fine-tuning.

    LoRA (Low-Rank Adaptation) adds small trainable matrices to specific layers
    instead of updating all model parameters. This drastically reduces memory
    and compute requirements.

    Args:
        r: Rank of LoRA decomposition (lower = fewer parameters, but less capacity).
        lora_alpha: Scaling factor for LoRA weights (typically 2*r or higher).
        lora_dropout: Dropout rate applied to LoRA layers during training.
        target_modules: List of module names to apply LoRA to.
                       Default: ["q_proj", "k_proj", "v_proj"] (attention projections).

    Returns:
        LoraConfig object ready to be applied to a model.
    """

    # Default target modules are the query, key, and value projections in attention layers
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj"]

    # Create LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # We're extracting embeddings, not generating text
        inference_mode=False,  # We will train, so not inference-only
        r=r,  # Rank of LoRA decomposition
        lora_alpha=lora_alpha,  # Scaling factor
        lora_dropout=lora_dropout,  # Dropout on LoRA layers
        target_modules=target_modules  # Which layers to apply LoRA to
    )

    return lora_config


def apply_lora_to_model(model: AutoModel, lora_config: LoraConfig) -> AutoModel:
    """
    Apply LoRA adapters to a model using PEFT.

    This wraps the base model with LoRA adapters. Only the adapter weights
    will be trainable, while the base model remains frozen.

    Args:
        model: The base EmbeddingGemma model (should be frozen first).
        lora_config: LoRA configuration created by configure_lora().

    Returns:
        PEFT model with LoRA adapters applied.
    """

    # Wrap the model with LoRA adapters
    # This creates new trainable parameters (the adapter matrices)
    # while keeping the base model weights frozen
    peft_model = get_peft_model(model, lora_config)

    return peft_model


def setup_lora_model(
    model: AutoModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> AutoModel:
    """
    Complete setup: freeze base model and apply LoRA adapters.

    This is a convenience function that combines freezing and LoRA application.

    Args:
        model: The base EmbeddingGemma model.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        target_modules: Modules to apply LoRA to.

    Returns:
        PEFT model with LoRA adapters, ready for training.
    """

    # Step 1: Freeze the base model
    model = freeze_base_model(model)

    # Step 2: Configure LoRA
    lora_config = configure_lora(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )

    # Step 3: Apply LoRA to the model
    peft_model = apply_lora_to_model(model, lora_config)

    return peft_model


def print_trainable_parameters(model: AutoModel) -> dict:
    """
    Print and return statistics about trainable vs total parameters.

    Args:
        model: The model to analyze (should be a PEFT model with LoRA).

    Returns:
        Dictionary with parameter counts and percentages.
    """

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count total parameters
    all_params = sum(p.numel() for p in model.parameters())

    # Calculate percentage
    trainable_percentage = 100 * trainable_params / all_params if all_params > 0 else 0

    # Print statistics
    print(f"Trainable params: {trainable_params:,} ({trainable_percentage:.2f}% of {all_params:,})")

    return {
        "trainable": trainable_params,
        "total": all_params,
        "percentage": trainable_percentage
    }

