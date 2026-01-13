"""
Used in: 07_Visualization_Embeddings.ipynb
Purpose:
    Provide utilities for visualizing high-dimensional embeddings in 2D space
    using dimensionality reduction techniques like PCA and t-SNE.
"""

import torch  # PyTorch for tensor operations
import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
from sklearn.decomposition import PCA  # PCA for dimensionality reduction
from sklearn.manifold import TSNE  # t-SNE for dimensionality reduction
from typing import List, Optional, Tuple  # Type hints


def reduce_to_2d_pca(embeddings: torch.Tensor, n_components: int = 2) -> np.ndarray:
    """
    Reduce embeddings to 2D using Principal Component Analysis (PCA).

    PCA finds the directions of maximum variance in the data and projects
    the embeddings onto the first two principal components.

    Args:
        embeddings: Tensor of shape (N, embedding_dim) with embeddings.
        n_components: Number of components (should be 2 for 2D visualization).

    Returns:
        NumPy array of shape (N, 2) with 2D coordinates.
    """

    # Convert to NumPy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings

    # Apply PCA
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(embeddings_np)

    return embeddings_2d


def reduce_to_2d_tsne(
    embeddings: torch.Tensor,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embeddings to 2D using t-SNE (t-distributed Stochastic Neighbor Embedding).

    t-SNE is a non-linear dimensionality reduction technique that preserves
    local neighborhood structure, making it good for visualizing clusters.

    Args:
        embeddings: Tensor of shape (N, embedding_dim) with embeddings.
        n_components: Number of components (should be 2 for 2D visualization).
        perplexity: Perplexity parameter (typically 5-50, lower for smaller datasets).
        random_state: Random seed for reproducibility.

    Returns:
        NumPy array of shape (N, 2) with 2D coordinates.
    """

    # Convert to NumPy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings

    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state
    )
    embeddings_2d = tsne.fit_transform(embeddings_np)

    return embeddings_2d


def plot_embeddings_2d(
    embeddings_2d: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    title: str = "2D Embedding Visualization",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot embeddings in 2D space.

    Args:
        embeddings_2d: NumPy array of shape (N, 2) with 2D coordinates.
        labels: Optional list of labels for each point.
        colors: Optional list of colors for each point.
        markers: Optional list of marker styles for each point.
        title: Plot title.
        figsize: Figure size (width, height).
        save_path: Optional path to save the plot.
    """

    # Create figure
    plt.figure(figsize=figsize)

    # Plot points
    if colors is None:
        # Default: all points same color
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    else:
        # Plot with different colors
        for i, (x, y) in enumerate(embeddings_2d):
            marker = markers[i] if markers and i < len(markers) else 'o'
            color = colors[i] if i < len(colors) else 'blue'
            plt.scatter(x, y, c=color, marker=marker, alpha=0.6, s=100)

    # Add labels if provided
    if labels:
        for i, (x, y) in enumerate(embeddings_2d):
            if i < len(labels):
                plt.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points')

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_paired_embeddings(
    anchor_embeddings_2d: np.ndarray,
    positive_embeddings_2d: np.ndarray,
    anchor_labels: Optional[List[str]] = None,
    positive_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: str = "2D Embedding Visualization (Anchors and Positives)",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot anchor and positive embeddings in 2D, showing pairs.

    Args:
        anchor_embeddings_2d: NumPy array of shape (N, 2) with anchor 2D coordinates.
        positive_embeddings_2d: NumPy array of shape (N, 2) with positive 2D coordinates.
        anchor_labels: Optional labels for anchor points.
        positive_labels: Optional labels for positive points.
        colors: Optional list of colors (one per pair).
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the plot.
    """

    # Create figure
    plt.figure(figsize=figsize)

    # Default colors if not provided
    if colors is None:
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Plot each pair
    num_pairs = len(anchor_embeddings_2d)
    for i in range(num_pairs):
        color = colors[i % len(colors)]

        # Plot anchor (X marker)
        plt.scatter(
            anchor_embeddings_2d[i, 0],
            anchor_embeddings_2d[i, 1],
            color=color,
            marker='X',
            s=150,
            alpha=0.8,
            label=f"Pair {i+1} anchor" if i == 0 else ""
        )

        # Plot positive (circle marker)
        plt.scatter(
            positive_embeddings_2d[i, 0],
            positive_embeddings_2d[i, 1],
            color=color,
            marker='o',
            s=150,
            alpha=0.8,
            label=f"Pair {i+1} positive" if i == 0 else ""
        )

        # Draw line connecting pair
        plt.plot(
            [anchor_embeddings_2d[i, 0], positive_embeddings_2d[i, 0]],
            [anchor_embeddings_2d[i, 1], positive_embeddings_2d[i, 1]],
            color=color,
            linestyle='--',
            alpha=0.3,
            linewidth=1
        )

        # Add labels if provided
        if anchor_labels and i < len(anchor_labels):
            plt.annotate(
                f"A{i+1}",
                (anchor_embeddings_2d[i, 0], anchor_embeddings_2d[i, 1]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        if positive_labels and i < len(positive_labels):
            plt.annotate(
                f"P{i+1}",
                (positive_embeddings_2d[i, 0], positive_embeddings_2d[i, 1]),
                xytext=(5, 5),
                textcoords='offset points'
            )

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

