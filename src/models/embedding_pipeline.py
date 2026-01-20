"""
Used in: 03_Model_Setup_Embeddings.ipynb, training, and inference workflows
Purpose:
    Provide the core embedding pipeline: loading EmbeddingGemma model,
    tokenization, forward pass, mean pooling, and normalization.
"""

import os  # Standard library env access for HF_TOKEN
import torch  # PyTorch for tensor operations and model inference
import torch.nn.functional as F  # Functional operations like normalization
from transformers import AutoTokenizer, AutoModel  # Hugging Face model loading
from typing import List, Union  # Type hints for function parameters


def load_embeddinggemma_model(model_name: str = "google/embeddinggemma-300m", device: str = None):
    """
    Load the EmbeddingGemma model and tokenizer from Hugging Face.

    Args:
        model_name: Hugging Face model identifier (default: google/embeddinggemma-300m)
        device: Device to load model on ('cuda', 'cpu', or None for auto-detection)

    Returns:
        Tuple of (tokenizer, model) ready for inference.
    """

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model (this will download weights if not cached)
    # Note: Make sure you've accepted the model license on Hugging Face
    # If HF_TOKEN is set in the environment, it is used for gated models.
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token)

    # Move model to the specified device
    model.to(device)
    model.eval()  # Set to evaluation mode (no dropout, etc.)

    return tokenizer, model


def embed_texts(
    texts: Union[str, List[str]],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = None,
    max_length: int = 128
) -> torch.Tensor:
    """
    Compute embeddings for a list of texts using the given model and tokenizer.

    This function handles the full pipeline:
    1. Tokenization with padding/truncation
    2. Model forward pass
    3. Mean pooling over token embeddings (weighted by attention mask)
    4. L2 normalization for cosine similarity

    Args:
        texts: Single string or list of strings to embed.
        model: The EmbeddingGemma model (or fine-tuned version with LoRA).
        tokenizer: The tokenizer matching the model.
        device: Device to run on (None for auto-detection).
        max_length: Maximum sequence length for tokenization.

    Returns:
        Tensor of shape (len(texts), 768) with normalized embeddings.
    """

    # Convert single string to list for uniform processing
    if isinstance(texts, str):
        texts = [texts]

    # Auto-detect device if not specified
    if device is None:
        device = next(model.parameters()).device

    # Tokenize the inputs; pad them to the same length and get attention masks
    # padding=True ensures all sequences are the same length
    # truncation=True cuts sequences longer than max_length
    # return_tensors="pt" returns PyTorch tensors
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Move tensors to the same device as the model
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Run through the model (no gradient needed for inference)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract hidden states from the last layer
    # Shape: (batch_size, sequence_length, hidden_size) = (N, seq_len, 768)
    token_embeddings = outputs.last_hidden_state

    # Mean pooling: sum the token vectors for each sequence, weighted by attention mask
    # Expand attention mask to match embedding dimensions
    mask_expand = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Sum embeddings, weighted by mask (padding tokens contribute 0)
    sum_embeddings = torch.sum(token_embeddings * mask_expand, dim=1)

    # Count actual tokens (not padding) for each sequence
    # Clamp to avoid division by zero (though this shouldn't happen with valid inputs)
    sum_mask = torch.clamp(mask_expand.sum(dim=1), min=1e-9)

    # Divide by token count to get mean
    pooled = sum_embeddings / sum_mask

    # Normalize the pooled embeddings to length 1 (for cosine similarity)
    # This makes dot product equal to cosine similarity
    pooled = F.normalize(pooled, p=2, dim=1)

    # Return CPU tensor for easier handling (can move back to GPU if needed)
    return pooled.cpu()


def compute_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix for a set of embeddings.

    Since embeddings are already normalized, cosine similarity = dot product.

    Args:
        embeddings: Tensor of shape (N, 768) with normalized embeddings.

    Returns:
        Tensor of shape (N, N) with pairwise cosine similarities.
    """

    # Since embeddings are normalized, dot product = cosine similarity
    similarity_matrix = embeddings @ embeddings.T

    return similarity_matrix

