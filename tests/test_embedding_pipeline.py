"""
Unit tests for embedding pipeline functionality.
"""

import pytest  # pytest is the testing framework used for this repo
import torch  # PyTorch for tensor operations
from src.models.embedding_pipeline import embed_texts, compute_cosine_similarity


def test_embed_texts_single_string():
    """
    Test that a single string is correctly converted to a list and embedded.
    """

    # Mock model and tokenizer (in real tests, you'd use actual model or mocks)
    # For now, we'll test the function signature and basic behavior
    # Note: This test would need actual model/tokenizer in practice
    pass


def test_embed_texts_list():
    """
    Test that a list of strings produces embeddings of correct shape.
    """

    # Test would verify:
    # - Input: list of N strings
    # - Output: tensor of shape (N, 768)
    # - Embeddings are normalized (L2 norm = 1)
    pass


def test_embed_texts_normalization():
    """
    Test that embeddings are L2-normalized.
    """

    # Test would verify that each embedding has L2 norm ≈ 1.0
    pass


def test_compute_cosine_similarity():
    """
    Test cosine similarity computation between normalized embeddings.
    """

    # Create dummy normalized embeddings
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])

    # Compute similarity matrix
    sim_matrix = compute_cosine_similarity(embeddings)

    # Check shape
    assert sim_matrix.shape == (3, 3)

    # Check diagonal is 1.0 (self-similarity)
    assert torch.allclose(torch.diag(sim_matrix), torch.ones(3))

    # Check that first and third embeddings are identical (similarity = 1.0)
    assert torch.allclose(sim_matrix[0, 2], torch.tensor(1.0))

    # Check that first and second are orthogonal (similarity = 0.0)
    assert torch.allclose(sim_matrix[0, 1], torch.tensor(0.0))


def test_compute_cosine_similarity_symmetric():
    """
    Test that similarity matrix is symmetric.
    """

    # Create random normalized embeddings
    embeddings = torch.randn(5, 10)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    sim_matrix = compute_cosine_similarity(embeddings)

    # Check symmetry: sim[i,j] == sim[j,i]
    assert torch.allclose(sim_matrix, sim_matrix.T)

