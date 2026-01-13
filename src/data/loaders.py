"""
Used in: 02_Load_Data_Explore.ipynb
Purpose:
    Provide standardized data-loading functions to keep notebooks clean and modular.
    Supports loading semantic similarity datasets for embedding fine-tuning.
"""

import pandas as pd  # Pandas is the main library for working with tabular data
from typing import Optional, List, Dict  # Type hints for function parameters
from pathlib import Path  # Robust path handling


def load_toy_dataset() -> List[Dict[str, str]]:
    """
    Load a toy semantic similarity dataset for demonstration.

    Returns:
        A list of dictionaries, each containing 'anchor' and 'positive' keys.
        Each pair represents semantically similar sentences.
    """

    # Toy dataset of similar sentence pairs
    # These pairs are designed to test the model's ability to learn semantic similarity
    train_data = [
        {"anchor": "I love playing football.", "positive": "Playing soccer is my favorite hobby."},
        {"anchor": "The weather is really nice today.", "positive": "It's quite a sunny day outside."},
        {"anchor": "The soccer game ended with no winner.", "positive": "The football match ended in a draw."},
        {"anchor": "There was a heavy downpour all day.", "positive": "It rained heavily throughout the day."}
    ]

    return train_data


def load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a CSV file as a DataFrame.

    Args:
        path: Path to the CSV file.
        nrows: Number of rows to read (None means read entire file).

    Returns:
        DataFrame containing the data.
    """

    # Use pandas to load the file.
    # nrows makes large files easier to preview or test.
    df = pd.read_csv(path, nrows=nrows)

    # Return the DataFrame to the caller
    return df


def load_similarity_pairs(path: str) -> List[Dict[str, str]]:
    """
    Load semantic similarity pairs from a CSV file.

    Expected CSV format:
        - Columns: 'anchor' and 'positive' (or 'query' and 'passage')
        - Each row represents a positive pair

    Args:
        path: Path to the CSV file containing similarity pairs.

    Returns:
        A list of dictionaries with 'anchor' and 'positive' keys.
    """

    # Load the CSV file
    df = pd.read_csv(path)

    # Handle different column name conventions
    if 'anchor' in df.columns and 'positive' in df.columns:
        anchor_col = 'anchor'
        positive_col = 'positive'
    elif 'query' in df.columns and 'passage' in df.columns:
        anchor_col = 'query'
        positive_col = 'passage'
    elif 'sentence1' in df.columns and 'sentence2' in df.columns:
        anchor_col = 'sentence1'
        positive_col = 'sentence2'
    else:
        raise ValueError(
            f"CSV must have columns ('anchor', 'positive'), ('query', 'passage'), "
            f"or ('sentence1', 'sentence2'). Found: {list(df.columns)}"
        )

    # Convert to list of dictionaries
    pairs = []
    for _, row in df.iterrows():
        pairs.append({
            "anchor": str(row[anchor_col]),
            "positive": str(row[positive_col])
        })

    return pairs


def load_query_passage_pairs(path: str) -> List[Dict[str, str]]:
    """
    Load query-passage pairs from a CSV file and map to anchor-positive format.

    This function is specifically designed for PDF-generated query-passage pairs.
    It maps 'query' → 'anchor' and 'passage' → 'positive' for compatibility
    with the existing training framework.

    Args:
        path: Path to the CSV file containing query-passage pairs.

    Returns:
        A list of dictionaries with 'anchor' and 'positive' keys (compatible with trainer).
    """

    # Load the CSV file
    df = pd.read_csv(path)

    # Check for required columns
    if 'query' not in df.columns or 'passage' not in df.columns:
        # Fall back to load_similarity_pairs if columns don't match
        return load_similarity_pairs(path)

    # Convert to list of dictionaries with anchor/positive mapping
    pairs = []
    for _, row in df.iterrows():
        pairs.append({
            "anchor": str(row['query']),
            "positive": str(row['passage'])
        })

    return pairs


def validate_pairs(pairs: List[Dict[str, str]]) -> Dict[str, any]:
    """
    Validate a list of similarity pairs and return statistics.

    Args:
        pairs: List of dictionaries with 'anchor' and 'positive' keys.

    Returns:
        Dictionary containing validation statistics:
        - count: Number of pairs
        - avg_anchor_length: Average character length of anchor sentences
        - avg_positive_length: Average character length of positive sentences
        - has_empty: Boolean indicating if any empty strings are present
    """

    if not pairs:
        return {
            "count": 0,
            "avg_anchor_length": 0,
            "avg_positive_length": 0,
            "has_empty": False
        }

    # Calculate statistics
    anchor_lengths = [len(pair.get("anchor", "")) for pair in pairs]
    positive_lengths = [len(pair.get("positive", "")) for pair in pairs]

    has_empty = any(
        not pair.get("anchor", "").strip() or not pair.get("positive", "").strip()
        for pair in pairs
    )

    return {
        "count": len(pairs),
        "avg_anchor_length": sum(anchor_lengths) / len(anchor_lengths) if anchor_lengths else 0,
        "avg_positive_length": sum(positive_lengths) / len(positive_lengths) if positive_lengths else 0,
        "has_empty": has_empty
    }

