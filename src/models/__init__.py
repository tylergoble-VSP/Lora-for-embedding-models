"""Model utilities for EmbeddingGemma and embedding computation."""

from src.models.embedding_pipeline import (
    load_embeddinggemma_model,
    embed_texts,
    compute_cosine_similarity
)
from src.models.lora_setup import (
    freeze_base_model,
    configure_lora,
    apply_lora_to_model,
    setup_lora_model,
    print_trainable_parameters
)

__all__ = [
    "load_embeddinggemma_model",
    "embed_texts",
    "compute_cosine_similarity",
    "freeze_base_model",
    "configure_lora",
    "apply_lora_to_model",
    "setup_lora_model",
    "print_trainable_parameters"
]

