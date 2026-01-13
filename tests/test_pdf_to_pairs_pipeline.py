"""
Smoke test for end-to-end PDF to training pairs pipeline.

Tests the complete pipeline with a small test PDF.
"""

import pytest  # Testing framework
import pandas as pd  # Pandas for DataFrame operations
from io import BytesIO  # In-memory file handling
from reportlab.lib.pagesizes import letter  # PDF page size
from reportlab.pdfgen import canvas  # PDF generation
from reportlab.lib.units import inch  # Unit conversion
import tempfile  # Temporary file handling
import os  # File operations

from src.pipelines.pdf_to_finetune_pairs import pdf_to_finetune_pairs


def create_simple_test_pdf() -> str:
    """
    Create a simple test PDF file.

    Returns:
        Path to the created PDF file.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Page 1
    c.drawString(1 * inch, 9 * inch, "1. Introduction")
    c.drawString(1 * inch, 8.5 * inch, "This document introduces machine learning concepts.")
    c.drawString(1 * inch, 8 * inch, "Machine learning is a subset of artificial intelligence.")
    c.showPage()

    # Page 2
    c.drawString(1 * inch, 9 * inch, "2. Methods")
    c.drawString(1 * inch, 8.5 * inch, "We use neural networks for classification.")
    c.drawString(1 * inch, 8 * inch, "The network consists of multiple layers.")
    c.showPage()

    c.save()
    buffer.seek(0)

    # Write to temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    tmp_file.write(buffer.read())
    tmp_file.close()

    return tmp_file.name


def test_pdf_to_finetune_pairs_smoke():
    """Smoke test for complete pipeline."""
    pdf_path = create_simple_test_pdf()

    try:
        # Run pipeline with minimal settings
        artifact_paths = pdf_to_finetune_pairs(
            pdf_path=pdf_path,
            doc_id="test_doc",
            max_tokens=128,  # Smaller for faster testing
            overlap_tokens=16,
            min_tokens=32,
            num_questions_per_chunk=2,  # Fewer questions for speed
            embed_chunks=False,  # Skip embedding for speed
            val_ratio=0.3
        )

        # Check that artifacts were created
        assert 'full_csv' in artifact_paths
        assert 'train_csv' in artifact_paths
        assert 'val_csv' in artifact_paths

        # Verify files exist
        assert os.path.exists(artifact_paths['full_csv'])
        assert os.path.exists(artifact_paths['train_csv'])
        assert os.path.exists(artifact_paths['val_csv'])

        # Load and verify data
        train_df = pd.read_csv(artifact_paths['train_csv'])
        val_df = pd.read_csv(artifact_paths['val_csv'])

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert 'query' in train_df.columns
        assert 'passage' in train_df.columns

        # Verify no data leakage (chunk_ids should not overlap)
        train_chunks = set(train_df['chunk_id'].unique())
        val_chunks = set(val_df['chunk_id'].unique())
        assert len(train_chunks & val_chunks) == 0  # No overlap

    finally:
        # Cleanup
        os.unlink(pdf_path)
        # Cleanup generated files (optional, can keep for inspection)
        # for path in artifact_paths.values():
        #     if os.path.exists(path):
        #         os.unlink(path)


def test_pdf_to_finetune_pairs_with_embedding():
    """Test pipeline with embedding enabled (slower but more complete)."""
    pytest.skip("Skipping embedding test to save time in CI - enable for full testing")

    pdf_path = create_simple_test_pdf()

    try:
        artifact_paths = pdf_to_finetune_pairs(
            pdf_path=pdf_path,
            doc_id="test_doc",
            max_tokens=128,
            overlap_tokens=16,
            min_tokens=32,
            num_questions_per_chunk=2,
            embed_chunks=True,  # Enable embedding
            embed_batch_size=4,  # Small batch for testing
            val_ratio=0.3
        )

        # Check embedding artifacts if embedding was enabled
        # (Note: embed_chunks saves internally, paths may not be in artifact_paths)
        assert 'full_csv' in artifact_paths

    finally:
        os.unlink(pdf_path)

