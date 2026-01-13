"""
Used in: 09_PDF_Ingest_Chunk_Embed.ipynb, pdf_to_finetune_pairs.py
Purpose:
    Embed text chunks using EmbeddingGemma with batching for efficiency.
    Saves embeddings and metadata to timestamped files.
"""

import numpy as np  # NumPy for array operations
import pandas as pd  # Pandas for DataFrame operations
import torch  # PyTorch for tensor operations
from pathlib import Path  # Path handling
from typing import Optional, Tuple  # Type hints

from src.models.embedding_pipeline import load_embeddinggemma_model, embed_texts  # Embedding functions
from src.utils.paths import timestamped_path  # Timestamped file paths
from src.utils.timing import TimeBlock  # Performance tracking


def embed_chunks(
    chunks_df: pd.DataFrame,
    batch_size: int = 64,
    max_length: int = 512,
    model_name: str = "google/embeddinggemma-300m",
    device: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Embed all chunks in a DataFrame using EmbeddingGemma with batching.

    This function:
    1. Loads the EmbeddingGemma model
    2. Processes chunks in batches for efficiency
    3. Saves embeddings to timestamped .npy file
    4. Saves metadata (chunk_id, doc_id, etc.) to timestamped parquet file

    Args:
        chunks_df: DataFrame with 'text' column and other metadata (chunk_id, doc_id, etc.).
        batch_size: Number of chunks to process in each batch.
        max_length: Maximum sequence length for tokenization (should match chunk max_tokens).
        model_name: Hugging Face model identifier.
        device: Device to run on ('cuda', 'cpu', or None for auto-detection).

    Returns:
        Tuple of (embeddings_array, metadata_df):
        - embeddings_array: NumPy array of shape (num_chunks, embedding_dim)
        - metadata_df: DataFrame with chunk metadata (chunk_id, doc_id, etc.)
    """

    with TimeBlock("embed_chunks"):
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading EmbeddingGemma model: {model_name}")
        print(f"Using device: {device}")

        # Load model and tokenizer
        tokenizer, model = load_embeddinggemma_model(model_name, device=device)

        # Extract text column
        texts = chunks_df['text'].tolist()
        num_chunks = len(texts)

        print(f"Embedding {num_chunks} chunks in batches of {batch_size}...")

        # Process in batches
        all_embeddings = []

        for i in range(0, num_chunks, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (num_chunks + batch_size - 1) // batch_size

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)...")

            # Embed this batch
            batch_embeddings = embed_texts(
                batch_texts,
                model,
                tokenizer,
                device=device,
                max_length=max_length
            )

            # Convert to NumPy and store
            all_embeddings.append(batch_embeddings.numpy())

        # Concatenate all batches
        embeddings_array = np.vstack(all_embeddings)

        print(f"Generated embeddings shape: {embeddings_array.shape}")

        # Create metadata DataFrame (copy relevant columns from chunks_df)
        metadata_columns = ['chunk_id', 'doc_id', 'section_id', 'section_title',
                           'page_start', 'page_end', 'token_count', 'char_count']
        available_columns = [col for col in metadata_columns if col in chunks_df.columns]
        metadata_df = chunks_df[available_columns].copy()

        # Ensure chunk_id is in metadata (required for retrieval)
        if 'chunk_id' not in metadata_df.columns and 'chunk_id' in chunks_df.index.name:
            metadata_df['chunk_id'] = chunks_df.index

        # Save embeddings to timestamped .npy file
        embeddings_path = timestamped_path("outputs/embeddings", "chunk_embeddings", "npy")
        np.save(embeddings_path, embeddings_array)
        print(f"Saved embeddings to: {embeddings_path}")

        # Save metadata to timestamped parquet file
        metadata_path = timestamped_path("outputs/embeddings", "chunk_embeddings_metadata", "parquet")
        metadata_df.to_parquet(metadata_path, index=False)
        print(f"Saved metadata to: {metadata_path}")

        return embeddings_array, metadata_df


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flat"
) -> Optional[object]:
    """
    Build a FAISS index for efficient nearest neighbor search (optional).

    This function is optional and requires faiss-cpu or faiss-gpu to be installed.

    Args:
        embeddings: NumPy array of shape (num_chunks, embedding_dim).
        index_type: Type of FAISS index ('flat' for exact search, 'ivf' for approximate).

    Returns:
        FAISS index object, or None if FAISS is not available.
    """

    try:
        import faiss  # FAISS for efficient similarity search

        # Normalize embeddings (FAISS works better with normalized vectors)
        embeddings_normalized = embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)

        # Build index based on type
        dimension = embeddings.shape[1]

        if index_type == "flat":
            # Flat index: exact search, no compression
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity for normalized)
        elif index_type == "ivf":
            # IVF index: approximate search with clustering
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings_normalized)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        # Add embeddings to index
        index.add(embeddings_normalized)

        print(f"Built FAISS {index_type} index with {index.ntotal} vectors")
        return index

    except ImportError:
        print("FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu)")
        return None


def build_sklearn_index(
    embeddings: np.ndarray,
    n_neighbors: int = 10
) -> Optional[object]:
    """
    Build a scikit-learn NearestNeighbors index for efficient nearest neighbor search (optional).

    This is a lightweight alternative to FAISS that doesn't require additional dependencies.

    Args:
        embeddings: NumPy array of shape (num_chunks, embedding_dim).
        n_neighbors: Number of neighbors to return in queries.

    Returns:
        Fitted NearestNeighbors object, or None if scikit-learn is not available.
    """

    try:
        from sklearn.neighbors import NearestNeighbors  # scikit-learn nearest neighbors

        # Build and fit the index
        index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        index.fit(embeddings)

        print(f"Built scikit-learn NearestNeighbors index with {len(embeddings)} vectors")
        return index

    except ImportError:
        print("scikit-learn not available. Install with: pip install scikit-learn")
        return None

