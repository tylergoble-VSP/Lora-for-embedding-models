"""
Used in: 05_Fine_Tuning_Training.ipynb, scripts/train_lora_embeddings.py
Purpose:
    Provide training loop implementation for fine-tuning EmbeddingGemma with LoRA
    using contrastive learning on semantic similarity pairs.
"""

import torch  # PyTorch for tensor operations and optimization
import torch.nn.functional as F  # Functional operations
from torch.optim import AdamW  # AdamW optimizer
from typing import List, Dict, Optional  # Type hints
from transformers import AutoModel, AutoTokenizer  # Model and tokenizer types

from src.models.embedding_pipeline import embed_texts  # Embedding computation
from src.training.contrastive_loss import multiple_negatives_ranking_loss  # Loss function
from src.utils.timing import timeit, TimeBlock  # Performance tracking


def prepare_batch(
    batch: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of anchor-positive pairs for training.

    Args:
        batch: List of dictionaries with 'anchor' and 'positive' keys.
        tokenizer: Tokenizer for encoding text.
        device: Device to place tensors on.

    Returns:
        Dictionary with tokenized inputs for anchors and positives.
    """

    # Extract anchor and positive sentences
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]

    # Tokenize anchors and positives
    enc_anch = tokenizer(
        anchors,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    enc_pos = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    return {
        "anchors": enc_anch,
        "positives": enc_pos
    }


def compute_embeddings_from_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute sentence embeddings from token hidden states using mean pooling.

    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim) from model.
        attention_mask: Tensor of shape (batch_size, seq_len) indicating real tokens.

    Returns:
        Tensor of shape (batch_size, hidden_dim) with normalized sentence embeddings.
    """

    # Expand attention mask to match hidden states dimensions
    mask_expand = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

    # Sum embeddings, weighted by mask (padding tokens contribute 0)
    sum_embeddings = torch.sum(hidden_states * mask_expand, dim=1)

    # Count actual tokens (not padding) for each sequence
    sum_mask = torch.clamp(mask_expand.sum(dim=1), min=1e-9)

    # Divide by token count to get mean
    pooled = sum_embeddings / sum_mask

    # Normalize to unit length for cosine similarity
    pooled = F.normalize(pooled, p=2, dim=1)

    return pooled


@timeit
def train_epoch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    train_data: List[Dict[str, str]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: str,
    temperature: float = 1.0
) -> float:
    """
    Train the model for one epoch using contrastive learning.

    Args:
        model: The model with LoRA adapters (in training mode).
        tokenizer: Tokenizer for encoding text.
        train_data: List of training pairs with 'anchor' and 'positive' keys.
        optimizer: Optimizer for updating parameters.
        batch_size: Number of pairs per batch.
        device: Device to run training on.
        temperature: Temperature scaling for loss (default 1.0).

    Returns:
        Average loss for the epoch.
    """

    model.train()  # Set model to training mode (enables dropout, etc.)

    total_loss = 0.0
    num_batches = 0

    # Process data in batches
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]

        # Prepare batch (tokenize anchors and positives)
        batch_data = prepare_batch(batch, tokenizer, device)

        # Forward pass: get embeddings for anchors and positives
        outputs_anch = model(**batch_data["anchors"])
        outputs_pos = model(**batch_data["positives"])

        # Extract hidden states from last layer
        hidden_anch = outputs_anch.last_hidden_state  # (batch, seq_len, 768)
        hidden_pos = outputs_pos.last_hidden_state

        # Compute sentence embeddings via mean pooling
        anch_emb = compute_embeddings_from_hidden_states(
            hidden_anch,
            batch_data["anchors"]["attention_mask"]
        )
        pos_emb = compute_embeddings_from_hidden_states(
            hidden_pos,
            batch_data["positives"]["attention_mask"]
        )

        # Compute contrastive loss
        loss = multiple_negatives_ranking_loss(
            anch_emb,
            pos_emb,
            temperature=temperature
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # Return average loss
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_model(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    train_data: List[Dict[str, str]],
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    temperature: float = 1.0,
    device: Optional[str] = None
) -> List[float]:
    """
    Train the model for multiple epochs.

    Args:
        model: The model with LoRA adapters.
        tokenizer: Tokenizer for encoding text.
        train_data: List of training pairs.
        epochs: Number of training epochs.
        batch_size: Number of pairs per batch.
        learning_rate: Learning rate for optimizer.
        temperature: Temperature scaling for loss.
        device: Device to run on (None for auto-detection).

    Returns:
        List of average losses per epoch.
    """

    # Auto-detect device if not specified
    if device is None:
        device = next(model.parameters()).device

    # Create optimizer (only trainable parameters, i.e., LoRA weights)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # Track losses
    losses = []

    # Training loop
    with TimeBlock("full_training"):
        for epoch in range(1, epochs + 1):
            avg_loss = train_epoch(
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                optimizer=optimizer,
                batch_size=batch_size,
                device=device,
                temperature=temperature
            )

            losses.append(avg_loss)
            print(f"Epoch {epoch}/{epochs}: training loss = {avg_loss:.4f}")

    return losses

