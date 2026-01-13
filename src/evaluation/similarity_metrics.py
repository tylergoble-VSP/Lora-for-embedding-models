"""
Used in: 06_Evaluation_Analysis.ipynb, visualization notebooks
Purpose:
    Provide metrics for evaluating embedding quality, including cosine similarity
    computation, ranking accuracy, and hard negative analysis.
"""

import torch  # PyTorch for tensor operations
import numpy as np  # NumPy for numerical operations
from typing import List, Dict, Tuple  # Type hints


def compute_similarity_matrix(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between two sets of embeddings.

    Args:
        embeddings1: Tensor of shape (N, embedding_dim) with normalized embeddings.
        embeddings2: Tensor of shape (M, embedding_dim) with normalized embeddings.

    Returns:
        Tensor of shape (N, M) with pairwise cosine similarities.
    """

    # Since embeddings are normalized, dot product = cosine similarity
    similarity_matrix = embeddings1 @ embeddings2.T

    return similarity_matrix


def rank_positives_for_anchors(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank positive embeddings by similarity to each anchor.

    Args:
        anchor_embeddings: Tensor of shape (N, embedding_dim) with anchor embeddings.
        positive_embeddings: Tensor of shape (M, embedding_dim) with positive embeddings.

    Returns:
        Tuple of (rankings, similarities):
        - rankings: Tensor of shape (N, M) with indices of positives sorted by similarity (descending).
        - similarities: Tensor of shape (N, M) with similarity scores.
    """

    # Compute similarity matrix
    similarities = compute_similarity_matrix(anchor_embeddings, positive_embeddings)

    # Get rankings (indices sorted by similarity, descending)
    rankings = torch.argsort(similarities, dim=1, descending=True)

    return rankings, similarities


def compute_accuracy(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor
) -> float:
    """
    Compute accuracy: percentage of anchors where the correct positive is ranked #1.

    Assumes anchor i should match with positive i (paired data).

    Args:
        anchor_embeddings: Tensor of shape (N, embedding_dim) with anchor embeddings.
        positive_embeddings: Tensor of shape (N, embedding_dim) with positive embeddings.

    Returns:
        Accuracy as a float between 0 and 1.
    """

    # Get rankings
    rankings, _ = rank_positives_for_anchors(anchor_embeddings, positive_embeddings)

    # Check if the correct positive (at index i) is ranked first for anchor i
    correct = 0
    for i in range(len(anchor_embeddings)):
        if rankings[i, 0].item() == i:  # Correct positive is ranked #1
            correct += 1

    accuracy = correct / len(anchor_embeddings)

    return accuracy


def compute_mean_reciprocal_rank(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR): average of 1/rank for correct positives.

    Args:
        anchor_embeddings: Tensor of shape (N, embedding_dim) with anchor embeddings.
        positive_embeddings: Tensor of shape (N, embedding_dim) with positive embeddings.

    Returns:
        MRR as a float between 0 and 1.
    """

    # Get rankings
    rankings, _ = rank_positives_for_anchors(anchor_embeddings, positive_embeddings)

    # Find rank of correct positive for each anchor
    reciprocal_ranks = []
    for i in range(len(anchor_embeddings)):
        # Find where the correct positive (index i) appears in rankings
        rank = (rankings[i] == i).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            rank_value = rank[0].item() + 1  # Convert to 1-based rank
            reciprocal_ranks.append(1.0 / rank_value)
        else:
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks)

    return mrr


def analyze_hard_negatives(
    anchor_texts: List[str],
    positive_texts: List[str],
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Identify hard negatives: incorrect positives with high similarity to anchors.

    Args:
        anchor_texts: List of anchor sentence strings.
        positive_texts: List of positive sentence strings.
        anchor_embeddings: Tensor of shape (N, embedding_dim) with anchor embeddings.
        positive_embeddings: Tensor of shape (M, embedding_dim) with positive embeddings.
        threshold: Similarity threshold above which to consider a pair a hard negative.

    Returns:
        List of dictionaries with hard negative pairs and their similarities.
    """

    # Compute similarity matrix
    similarities = compute_similarity_matrix(anchor_embeddings, positive_embeddings)

    hard_negatives = []

    # Check each anchor against all positives
    for i, anchor_text in enumerate(anchor_texts):
        for j, positive_text in enumerate(positive_texts):
            # Skip the correct positive pair
            if i == j:
                continue

            sim = similarities[i, j].item()

            # If similarity is above threshold, it's a hard negative
            if sim >= threshold:
                hard_negatives.append({
                    "anchor": anchor_text,
                    "incorrect_positive": positive_text,
                    "similarity": sim,
                    "anchor_idx": i,
                    "positive_idx": j
                })

    # Sort by similarity (descending)
    hard_negatives.sort(key=lambda x: x["similarity"], reverse=True)

    return hard_negatives

