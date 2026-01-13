"""
Unit tests for LoRA setup functionality.
"""

import pytest  # pytest is the testing framework used for this repo
import torch  # PyTorch for tensor operations
from src.models.lora_setup import (
    freeze_base_model,
    configure_lora,
    print_trainable_parameters
)


def test_freeze_base_model():
    """
    Test that all parameters in base model are frozen (requires_grad=False).
    """

    # Create a simple model for testing
    model = torch.nn.Linear(10, 5)

    # Initially, parameters should be trainable
    assert all(p.requires_grad for p in model.parameters())

    # Freeze the model
    frozen_model = freeze_base_model(model)

    # All parameters should now be frozen
    assert all(not p.requires_grad for p in frozen_model.parameters())


def test_configure_lora_defaults():
    """
    Test LoRA configuration with default parameters.
    """

    lora_config = configure_lora()

    # Check default values
    assert lora_config.r == 16
    assert lora_config.lora_alpha == 32
    assert lora_config.lora_dropout == 0.1
    assert "q_proj" in lora_config.target_modules
    assert "k_proj" in lora_config.target_modules
    assert "v_proj" in lora_config.target_modules


def test_configure_lora_custom():
    """
    Test LoRA configuration with custom parameters.
    """

    custom_targets = ["custom_proj"]
    lora_config = configure_lora(
        r=8,
        lora_alpha=16,
        lora_dropout=0.2,
        target_modules=custom_targets
    )

    assert lora_config.r == 8
    assert lora_config.lora_alpha == 16
    assert lora_config.lora_dropout == 0.2
    assert lora_config.target_modules == custom_targets


def test_print_trainable_parameters():
    """
    Test trainable parameter counting and printing.
    """

    # Create a simple model
    model = torch.nn.Linear(10, 5)

    # All parameters should be trainable initially
    stats = print_trainable_parameters(model)
    assert stats["trainable"] == stats["total"]
    assert stats["percentage"] == 100.0

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # No parameters should be trainable
    stats = print_trainable_parameters(model)
    assert stats["trainable"] == 0
    assert stats["percentage"] == 0.0

