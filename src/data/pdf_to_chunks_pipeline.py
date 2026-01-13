"""
Used in: 09_PDF_Ingest_Chunk_Embed.ipynb, scripts/build_pdf_finetune_dataset.py
Purpose:
    Orchestrate PDF ingestion, cleaning, section detection, and chunking.
    Saves chunks to timestamped parquet files with metadata.
"""

import pandas as pd  # Pandas for DataFrame operations
import json  # JSON for saving metadata sidecars
from pathlib import Path  # Path handling
from typing import Optional, Dict, Any  # Type hints
from transformers import PreTrainedTokenizer  # Tokenizer type

from src.data.pdf_ingest import PDFPage  # PDFPage dataclass
from src.data.text_cleaning import load_and_clean_pdf  # PDF cleaning
from src.data.chunking import build_sections, chunk_sections, TextChunk  # Chunking functions
from src.models.embedding_pipeline import load_embeddinggemma_model  # For tokenizer
from src.utils.paths import timestamped_path  # Timestamped file paths
from src.utils.timing import TimeBlock  # Performance tracking


def pdf_to_chunks(
    pdf_path: str,
    doc_id: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 128,
    min_heading_score: float = 0.4
) -> pd.DataFrame:
    """
    Complete pipeline: PDF → pages → sections → chunks → DataFrame.

    This function:
    1. Loads and cleans PDF pages
    2. Detects sections based on headings
    3. Chunks sections with token-aware packing
    4. Saves chunks to timestamped parquet file
    5. Saves JSON sidecar with config and statistics

    Args:
        pdf_path: Path to the PDF file.
        doc_id: Document identifier (defaults to PDF filename without extension).
        tokenizer: Tokenizer for token counting (defaults to EmbeddingGemma tokenizer).
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of tokens to overlap between chunks.
        min_tokens: Minimum tokens per chunk.
        min_heading_score: Minimum confidence score for heading detection.

    Returns:
        DataFrame with columns: doc_id, chunk_id, section_id, section_title,
        page_start, page_end, token_count, char_count, text.
    """

    with TimeBlock("pdf_to_chunks_pipeline"):
        # Generate doc_id from PDF filename if not provided
        if doc_id is None:
            doc_id = Path(pdf_path).stem

        # Load tokenizer if not provided
        if tokenizer is None:
            _, model = load_embeddinggemma_model()
            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
            # If model doesn't have tokenizer attribute, load it separately
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

        # Step 1: Load and clean PDF
        pages = load_and_clean_pdf(pdf_path)

        # Step 2: Build sections from pages
        sections = build_sections(pages, pdf_path, min_heading_score=min_heading_score)

        # Step 3: Chunk sections
        chunks = chunk_sections(
            sections,
            tokenizer,
            doc_id,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens
        )

        # Step 4: Convert chunks to DataFrame
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                'doc_id': chunk.doc_id,
                'chunk_id': chunk.chunk_id,
                'section_id': chunk.section_id,
                'section_title': chunk.section_title,
                'page_start': chunk.page_start,
                'page_end': chunk.page_end,
                'token_count': chunk.token_count,
                'char_count': chunk.char_count,
                'text': chunk.text
            })

        df = pd.DataFrame(chunk_data)

        # Step 5: Compute statistics
        # Convert NumPy/pandas types to native Python types for JSON serialization
        stats = {
            'num_pages': int(len(pages)),
            'num_sections': int(len(sections)),
            'num_chunks': int(len(chunks)),
            'total_tokens': int(df['token_count'].sum()) if len(df) > 0 else 0,
            'avg_tokens_per_chunk': float(df['token_count'].mean()) if len(df) > 0 else 0.0,
            'min_tokens': int(df['token_count'].min()) if len(df) > 0 else 0,
            'median_tokens': float(df['token_count'].median()) if len(df) > 0 else 0.0,
            'max_tokens': int(df['token_count'].max()) if len(df) > 0 else 0,
        }

        # Step 6: Save to parquet (timestamped)
        parquet_path = timestamped_path("data/processed", "pdf_chunks", "parquet")
        df.to_parquet(parquet_path, index=False)

        # Step 7: Save JSON sidecar with config and stats
        sidecar_data = {
            'pdf_path': pdf_path,
            'doc_id': doc_id,
            'config': {
                'max_tokens': max_tokens,
                'overlap_tokens': overlap_tokens,
                'min_tokens': min_tokens,
                'min_heading_score': min_heading_score
            },
            'statistics': stats,
            'parquet_path': str(parquet_path)
        }

        sidecar_path = timestamped_path("data/processed", "pdf_chunks_metadata", "json")
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)

        print(f"Saved {len(chunks)} chunks to: {parquet_path}")
        print(f"Saved metadata to: {sidecar_path}")
        print(f"Statistics: {stats}")

        return df

