"""
Used in: 06_Evaluation_Analysis.ipynb, 08_Semantic_Search_Demo.ipynb
Purpose:
    Provide evaluation utilities for semantic search and retrieval tasks,
    including top-k retrieval and retrieval metrics.
"""

import torch  # PyTorch for tensor operations
import numpy as np  # NumPy for numerical operations
from typing import List, Tuple, Dict  # Type hints


def retrieve_top_k(
    query_embedding: torch.Tensor,
    document_embeddings: torch.Tensor,
    k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve top-k most similar documents for a query.

    Args:
        query_embedding: Tensor of shape (embedding_dim,) with query embedding.
        document_embeddings: Tensor of shape (N, embedding_dim) with document embeddings.
        k: Number of top documents to retrieve.

    Returns:
        Tuple of (indices, similarities):
        - indices: Tensor of shape (k,) with indices of top-k documents.
        - similarities: Tensor of shape (k,) with similarity scores.
    """

    # Compute similarities between query and all documents
    # query_embedding: (embedding_dim,)
    # document_embeddings: (N, embedding_dim)
    # Result: (N,)
    similarities = query_embedding @ document_embeddings.T

    # Get top-k indices and values
    top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))

    return top_k_indices, top_k_values


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
    ground_truth: List[List[int]]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance using multiple metrics.

    Args:
        query_embeddings: Tensor of shape (N, embedding_dim) with query embeddings.
        document_embeddings: Tensor of shape (M, embedding_dim) with document embeddings.
        ground_truth: List of lists, where ground_truth[i] contains indices of relevant documents for query i.

    Returns:
        Dictionary with metrics:
        - precision_at_1: Percentage of queries with correct document at rank 1
        - precision_at_5: Average precision@5
        - mean_reciprocal_rank: MRR across all queries
    """

    num_queries = len(query_embeddings)
    precision_at_1 = 0.0
    precision_at_5 = 0.0
    reciprocal_ranks = []

    for i in range(num_queries):
        # Get top-5 retrieved documents
        top_indices, _ = retrieve_top_k(query_embeddings[i], document_embeddings, k=5)

        # Get relevant documents for this query
        relevant = set(ground_truth[i])

        # Precision@1: Is the top document relevant?
        if len(top_indices) > 0 and top_indices[0].item() in relevant:
            precision_at_1 += 1.0

        # Precision@5: How many of top-5 are relevant?
        relevant_in_top5 = sum(1 for idx in top_indices if idx.item() in relevant)
        precision_at_5 += relevant_in_top5 / min(5, len(top_indices))

        # MRR: Find rank of first relevant document
        for rank, idx in enumerate(top_indices, 1):
            if idx.item() in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant document found in top-5
            reciprocal_ranks.append(0.0)

    # Average metrics
    precision_at_1 /= num_queries
    precision_at_5 /= num_queries
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {
        "precision_at_1": precision_at_1,
        "precision_at_5": precision_at_5,
        "mean_reciprocal_rank": mrr
    }

