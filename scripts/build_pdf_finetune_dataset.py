#!/usr/bin/env python3
"""
CLI script for building PDF fine-tuning datasets.

This script orchestrates the complete pipeline:
PDF → chunks → embeddings → questions → training pairs

Usage:
    python scripts/build_pdf_finetune_dataset.py --pdf_path path/to/document.pdf
"""

import argparse  # Command-line argument parsing
import sys  # System operations
from pathlib import Path  # Path handling

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.pdf_to_finetune_pairs import pdf_to_finetune_pairs  # Main pipeline
from src.llm.question_generation import QuestionGenConfig  # Question generation config


def main():
    """
    Main entry point for the CLI script.

    Parses command-line arguments and runs the PDF to fine-tuning pairs pipeline.
    """

    parser = argparse.ArgumentParser(
        description="Build fine-tuning dataset from PDF: PDF → chunks → embeddings → questions → pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python scripts/build_pdf_finetune_dataset.py --pdf_path document.pdf

  # Custom chunking parameters
  python scripts/build_pdf_finetune_dataset.py --pdf_path document.pdf \\
      --max_tokens 256 --overlap_tokens 32 --min_tokens 64

  # Custom question generation
  python scripts/build_pdf_finetune_dataset.py --pdf_path document.pdf \\
      --questions_per_chunk 5 --question_model_name microsoft/DialoGPT-medium

  # Skip embedding step (faster, only builds dataset)
  python scripts/build_pdf_finetune_dataset.py --pdf_path document.pdf --no_embed_chunks
        """
    )

    # Required arguments
    parser.add_argument(
        '--pdf_path',
        type=str,
        required=True,
        help='Path to the PDF file to process'
    )

    # Optional arguments: document identification
    parser.add_argument(
        '--doc_id',
        type=str,
        default=None,
        help='Document identifier (defaults to PDF filename without extension)'
    )

    # Optional arguments: chunking parameters
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=512,
        help='Maximum tokens per chunk (default: 512)'
    )
    parser.add_argument(
        '--overlap_tokens',
        type=int,
        default=64,
        help='Number of tokens to overlap between chunks (default: 64)'
    )
    parser.add_argument(
        '--min_tokens',
        type=int,
        default=128,
        help='Minimum tokens per chunk (default: 128)'
    )

    # Optional arguments: question generation
    parser.add_argument(
        '--questions_per_chunk',
        type=int,
        default=3,
        help='Number of questions to generate per chunk (default: 3)'
    )
    parser.add_argument(
        '--question_model_name',
        type=str,
        default=None,
        help='HuggingFace model name for question generation (default: microsoft/DialoGPT-medium)'
    )

    # Optional arguments: embedding
    parser.add_argument(
        '--embed_batch_size',
        type=int,
        default=64,
        help='Batch size for embedding chunks (default: 64)'
    )
    parser.add_argument(
        '--no_embed_chunks',
        action='store_true',
        help='Skip chunk embedding step (faster, only builds dataset)'
    )

    # Optional arguments: train/val split
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Ratio of data for validation split (default: 0.2)'
    )

    # Optional arguments: output
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: uses timestamped paths in data/processed/)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    # Validate val_ratio
    if not 0.0 < args.val_ratio < 1.0:
        print(f"Error: val_ratio must be between 0.0 and 1.0, got {args.val_ratio}")
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print("PDF Fine-Tuning Dataset Builder")
    print("=" * 60)
    print(f"PDF Path: {pdf_path}")
    print(f"Document ID: {args.doc_id or pdf_path.stem}")
    print(f"\nChunking Parameters:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Overlap tokens: {args.overlap_tokens}")
    print(f"  Min tokens: {args.min_tokens}")
    print(f"\nQuestion Generation:")
    print(f"  Questions per chunk: {args.questions_per_chunk}")
    print(f"  Model: {args.question_model_name or 'default'}")
    print(f"\nEmbedding:")
    print(f"  Embed chunks: {not args.no_embed_chunks}")
    if not args.no_embed_chunks:
        print(f"  Batch size: {args.embed_batch_size}")
    print(f"\nTrain/Val Split:")
    print(f"  Val ratio: {args.val_ratio}")
    print("=" * 60)
    print()

    # Create question generation config if model name provided
    question_gen_config = None
    if args.question_model_name:
        question_gen_config = QuestionGenConfig(
            model_name=args.question_model_name,
            num_questions_per_chunk=args.questions_per_chunk
        )

    # Run pipeline
    try:
        artifact_paths = pdf_to_finetune_pairs(
            pdf_path=str(pdf_path),
            doc_id=args.doc_id,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            min_tokens=args.min_tokens,
            num_questions_per_chunk=args.questions_per_chunk,
            question_model_name=args.question_model_name,
            embed_batch_size=args.embed_batch_size,
            val_ratio=args.val_ratio,
            embed_chunks=not args.no_embed_chunks,
            question_gen_config=question_gen_config
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        for key, path in artifact_paths.items():
            print(f"  {key}: {path}")

        print("\nNext steps:")
        print("  1. Review the generated query-passage pairs")
        print("  2. Use the train/val CSV files for fine-tuning")
        print("  3. See notebook 11_LoRA_FineTune_EmbeddingGemma_on_PDF_QA.ipynb for training")

    except Exception as e:
        print(f"\nError: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

