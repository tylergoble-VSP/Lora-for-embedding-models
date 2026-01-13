"""
Unit tests for similarity metrics and evaluation functions.
"""

import pytest  # pytest is the testing framework used for this repo
import torch  # PyTorch for tensor operations
from src.evaluation.similarity_metrics import (
    compute_accuracy,
    compute_mean_reciprocal_rank
)


def test_compute_accuracy_perfect():
    """
    Test accuracy computation when all anchors match their correct positives.
    """

    # Create embeddings where anchor i is identical to positive i
    batch_size = 4
    embedding_dim = 10

    # Create identical anchor and positive embeddings
    base_embeddings = torch.randn(batch_size, embedding_dim)
    base_embeddings = torch.nn.functional.normalize(base_embeddings, p=2, dim=1)

    anchor_embeddings = base_embeddings.clone()
    positive_embeddings = base_embeddings.clone()

    # Accuracy should be 100% (each anchor matches its positive)
    accuracy = compute_accuracy(anchor_embeddings, positive_embeddings)

    assert accuracy == 1.0


def test_compute_accuracy_zero():
    """
    Test accuracy when no anchors match their correct positives.
    """

    batch_size = 3
    embedding_dim = 10

    # Create orthogonal embeddings (completely different)
    anchor_embeddings = torch.eye(batch_size, embedding_dim)
    positive_embeddings = torch.roll(torch.eye(batch_size, embedding_dim), shifts=1, dims=0)

    # Normalize
    anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
    positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)

    # Accuracy should be 0% (no matches)
    accuracy = compute_accuracy(anchor_embeddings, positive_embeddings)

    assert accuracy == 0.0


def test_compute_mean_reciprocal_rank_perfect():
    """
    Test MRR when all correct positives are ranked #1.
    """

    batch_size = 4
    embedding_dim = 10

    # Create identical embeddings
    base_embeddings = torch.randn(batch_size, embedding_dim)
    base_embeddings = torch.nn.functional.normalize(base_embeddings, p=2, dim=1)

    anchor_embeddings = base_embeddings.clone()
    positive_embeddings = base_embeddings.clone()

    # MRR should be 1.0 (all ranks are 1, so 1/1 = 1.0)
    mrr = compute_mean_reciprocal_rank(anchor_embeddings, positive_embeddings)

    assert abs(mrr - 1.0) < 1e-5


def test_compute_mean_reciprocal_rank_range():
    """
    Test that MRR is always between 0 and 1.
    """

    batch_size = 5
    embedding_dim = 10

    # Create random embeddings
    anchor_embeddings = torch.randn(batch_size, embedding_dim)
    anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)

    positive_embeddings = torch.randn(batch_size, embedding_dim)
    positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)

    mrr = compute_mean_reciprocal_rank(anchor_embeddings, positive_embeddings)

    assert 0.0 <= mrr <= 1.0

