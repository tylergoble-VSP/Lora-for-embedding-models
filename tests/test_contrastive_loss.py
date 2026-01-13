"""
Unit tests for contrastive loss computation.
"""

import pytest  # pytest is the testing framework used for this repo
import torch  # PyTorch for tensor operations
from src.training.contrastive_loss import multiple_negatives_ranking_loss


def test_multiple_negatives_ranking_loss_shape():
    """
    Test that loss function returns a scalar tensor.
    """

    # Create dummy normalized embeddings
    batch_size = 4
    embedding_dim = 768

    anchor_embeddings = torch.randn(batch_size, embedding_dim)
    anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)

    positive_embeddings = torch.randn(batch_size, embedding_dim)
    positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)

    # Compute loss
    loss = multiple_negatives_ranking_loss(anchor_embeddings, positive_embeddings)

    # Loss should be a scalar
    assert loss.shape == torch.Size([])
    assert isinstance(loss.item(), float)


def test_multiple_negatives_ranking_loss_positive():
    """
    Test that loss is positive (cross-entropy is always positive).
    """

    batch_size = 3
    embedding_dim = 10

    anchor_embeddings = torch.randn(batch_size, embedding_dim)
    anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)

    positive_embeddings = torch.randn(batch_size, embedding_dim)
    positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)

    loss = multiple_negatives_ranking_loss(anchor_embeddings, positive_embeddings)

    assert loss.item() > 0


def test_multiple_negatives_ranking_loss_temperature():
    """
    Test that temperature scaling affects the loss value.
    """

    batch_size = 2
    embedding_dim = 10

    anchor_embeddings = torch.randn(batch_size, embedding_dim)
    anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)

    positive_embeddings = torch.randn(batch_size, embedding_dim)
    positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)

    # Loss with temperature 1.0
    loss_t1 = multiple_negatives_ranking_loss(anchor_embeddings, positive_embeddings, temperature=1.0)

    # Loss with temperature 0.5 (should be different)
    loss_t05 = multiple_negatives_ranking_loss(anchor_embeddings, positive_embeddings, temperature=0.5)

    # Losses should be different (temperature affects softmax sharpness)
    assert not torch.allclose(loss_t1, loss_t05)

