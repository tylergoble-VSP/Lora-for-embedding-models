"""
Used in: 05_Fine_Tuning_Training.ipynb, trainer.py
Purpose:
    Implement Multiple Negatives Ranking Loss (MNR) for contrastive learning
    of embeddings. This loss encourages anchors to be similar to their positives
    and dissimilar to other positives in the batch (in-batch negatives).
"""

import torch  # PyTorch for tensor operations
import torch.nn.functional as F  # Functional operations like cross-entropy


def multiple_negatives_ranking_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute Multiple Negatives Ranking Loss (MNR) for contrastive learning.

    This loss function:
    1. Computes cosine similarity between each anchor and all positives
    2. Treats each anchor's own positive as the correct match
    3. Uses cross-entropy to maximize similarity to correct positive
    4. Other positives in the batch serve as negatives (in-batch negatives)

    Args:
        anchor_embeddings: Tensor of shape (batch_size, embedding_dim) with normalized anchor embeddings.
        positive_embeddings: Tensor of shape (batch_size, embedding_dim) with normalized positive embeddings.
        temperature: Temperature scaling factor (higher = sharper softmax). Default 1.0.

    Returns:
        Scalar tensor containing the loss value.
    """

    # Compute cosine similarity matrix
    # Since embeddings are normalized, dot product = cosine similarity
    # Shape: (batch_size, batch_size)
    similarity_matrix = anchor_embeddings @ positive_embeddings.T

    # Apply temperature scaling (optional, can sharpen the softmax)
    if temperature != 1.0:
        similarity_matrix = similarity_matrix / temperature

    # Create labels: each anchor i should match with positive i
    # Labels are [0, 1, 2, ..., batch_size-1]
    batch_size = similarity_matrix.size(0)
    labels = torch.arange(batch_size, device=similarity_matrix.device)

    # Cross-entropy loss: for each anchor, maximize similarity to its own positive
    # The loss treats similarity scores as logits in a classification problem
    # where the correct class is the matching positive
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


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

