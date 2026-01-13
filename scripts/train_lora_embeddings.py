#!/usr/bin/env python3
"""
Used in: Command-line training script (extracted from 05_Fine_Tuning_Training.ipynb)
Purpose:
    CLI script for training LoRA fine-tuned embedding models.
    Provides command-line interface for training without using notebooks.
"""

import argparse  # For command-line argument parsing
import sys  # For system operations
from pathlib import Path  # Path handling

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_toy_dataset, load_similarity_pairs  # Data loading
from src.models.embedding_pipeline import load_embeddinggemma_model  # Model loading
from src.models.lora_setup import setup_lora_model, print_trainable_parameters  # LoRA setup
from src.training.trainer import train_model  # Training loop
from src.utils.paths import timestamped_path  # Timestamped paths


def main():
    """
    Main function for CLI training script.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train LoRA fine-tuned EmbeddingGemma model for semantic similarity"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="toy",
        help="Dataset to use: 'toy' for toy dataset, or path to CSV file with 'anchor' and 'positive' columns"
    )

    # LoRA hyperparameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate (default: 0.1)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Loss temperature (default: 1.0)"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saved model (default: timestamped path in outputs/models/)"
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    if args.dataset == "toy":
        train_data = load_toy_dataset()
    else:
        # Load from CSV file
        train_data = load_similarity_pairs(args.dataset)

    print(f"Loaded {len(train_data)} training pairs")

    # Load model
    print("Loading EmbeddingGemma model...")
    tokenizer, base_model = load_embeddinggemma_model()

    # Setup LoRA
    print("Configuring LoRA...")
    model = setup_lora_model(
        base_model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj"]
    )

    print_trainable_parameters(model)

    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    losses = train_model(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature
    )

    print(f"\nTraining complete!")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

    # Save model
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = timestamped_path("outputs/models", "embeddinggemma_lora", "").parent

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to: {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print("Model saved successfully!")


if __name__ == "__main__":
    main()

