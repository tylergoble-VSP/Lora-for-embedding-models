"""
Used in: 08_Semantic_Search_Demo.ipynb
Purpose:
    Provide semantic search functionality using fine-tuned embeddings.
    Given a query, find the most relevant documents from a collection.
"""

import torch  # PyTorch for tensor operations
from typing import List, Tuple, Dict  # Type hints
from src.evaluation.retrieval_eval import retrieve_top_k  # Top-k retrieval


def embed_document_collection(
    documents: List[str],
    model,
    tokenizer
) -> torch.Tensor:
    """
    Embed a collection of documents for semantic search.

    Args:
        documents: List of document strings to embed.
        model: The embedding model (with or without LoRA).
        tokenizer: Tokenizer matching the model.

    Returns:
        Tensor of shape (len(documents), embedding_dim) with document embeddings.
    """

    from src.models.embedding_pipeline import embed_texts

    # Compute embeddings for all documents
    document_embeddings = embed_texts(documents, model, tokenizer)

    return document_embeddings


def search(
    query: str,
    document_embeddings: torch.Tensor,
    documents: List[str],
    model,
    tokenizer,
    top_k: int = 5
) -> List[Dict[str, any]]:
    """
    Perform semantic search: find top-k most relevant documents for a query.

    Args:
        query: Query string to search for.
        document_embeddings: Tensor of shape (N, embedding_dim) with pre-computed document embeddings.
        documents: List of document strings (must match document_embeddings order).
        model: The embedding model.
        tokenizer: Tokenizer matching the model.
        top_k: Number of top results to return.

    Returns:
        List of dictionaries, each containing:
        - 'document': The document text
        - 'similarity': Cosine similarity score
        - 'rank': Rank (1-based)
        - 'index': Original index in documents list
    """

    from src.models.embedding_pipeline import embed_texts

    # Embed the query
    query_embedding = embed_texts(query, model, tokenizer)

    # Retrieve top-k documents
    top_indices, top_similarities = retrieve_top_k(
        query_embedding.squeeze(0),  # Remove batch dimension
        document_embeddings,
        k=top_k
    )

    # Build results
    results = []
    for rank, (idx, sim) in enumerate(zip(top_indices, top_similarities), 1):
        idx_val = idx.item()
        sim_val = sim.item()
        results.append({
            'rank': rank,
            'document': documents[idx_val],
            'similarity': sim_val,
            'index': idx_val
        })

    return results


def search_batch(
    queries: List[str],
    document_embeddings: torch.Tensor,
    documents: List[str],
    model,
    tokenizer,
    top_k: int = 5
) -> List[List[Dict[str, any]]]:
    """
    Perform semantic search for multiple queries.

    Args:
        queries: List of query strings.
        document_embeddings: Tensor of shape (N, embedding_dim) with document embeddings.
        documents: List of document strings.
        model: The embedding model.
        tokenizer: Tokenizer matching the model.
        top_k: Number of top results per query.

    Returns:
        List of result lists (one per query), each containing top-k results.
    """

    results_batch = []
    for query in queries:
        results = search(
            query,
            document_embeddings,
            documents,
            model,
            tokenizer,
            top_k=top_k
        )
        results_batch.append(results)

    return results_batch

