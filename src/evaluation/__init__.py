"""Evaluation utilities for embedding quality and retrieval performance."""

from src.evaluation.similarity_metrics import (
    compute_similarity_matrix,
    rank_positives_for_anchors,
    compute_accuracy,
    compute_mean_reciprocal_rank,
    analyze_hard_negatives
)
from src.evaluation.retrieval_eval import (
    retrieve_top_k,
    evaluate_retrieval
)

__all__ = [
    "compute_similarity_matrix",
    "rank_positives_for_anchors",
    "compute_accuracy",
    "compute_mean_reciprocal_rank",
    "analyze_hard_negatives",
    "retrieve_top_k",
    "evaluate_retrieval"
]

