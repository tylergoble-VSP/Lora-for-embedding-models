"""
Used in: 10_Generate_Questions_Build_Dataset.ipynb, pdf_to_finetune_pairs.py
Purpose:
    Build training datasets of (query, passage) pairs from chunks and generated questions.
    Handles train/val splits with grouped splitting to avoid data leakage.
"""

import pandas as pd  # Pandas for DataFrame operations
from pathlib import Path  # Path handling
from typing import List, Tuple, Optional  # Type hints
from sklearn.model_selection import train_test_split  # Train/val splitting

from src.llm.question_generation import BaseQuestionGenerator, get_question_generator, QuestionGenConfig  # Question generation
from src.utils.paths import timestamped_path  # Timestamped file paths
from src.utils.timing import TimeBlock  # Performance tracking


def generate_questions_dataset(
    chunks_df: pd.DataFrame,
    question_generator: Optional[BaseQuestionGenerator] = None,
    num_questions_per_chunk: int = 3,
    question_gen_config: Optional[QuestionGenConfig] = None
) -> pd.DataFrame:
    """
    Generate (query, passage) pairs from chunks using question generation.

    This function:
    1. Generates questions for each chunk using the question generator
    2. Creates pairs of (question, chunk_text) for training
    3. Preserves metadata (doc_id, chunk_id, section_title, etc.)

    Args:
        chunks_df: DataFrame with 'text' column and metadata (chunk_id, doc_id, etc.).
        question_generator: Question generator instance (None creates default).
        num_questions_per_chunk: Number of questions to generate per chunk.
        question_gen_config: Config for question generator (if creating new one).

    Returns:
        DataFrame with columns: query, passage, doc_id, chunk_id, section_title,
        page_start, page_end, and other metadata.
    """

    with TimeBlock("generate_questions_dataset"):
        # Get or create question generator
        if question_generator is None:
            if question_gen_config is None:
                question_gen_config = QuestionGenConfig(
                    num_questions_per_chunk=num_questions_per_chunk
                )
            question_generator = get_question_generator(question_gen_config)

        # Generate questions for each chunk
        pairs_data = []
        num_chunks = len(chunks_df)

        print(f"Generating questions for {num_chunks} chunks...")

        for idx, row in chunks_df.iterrows():
            chunk_text = row['text']
            chunk_id = row.get('chunk_id', f'chunk_{idx}')

            # Generate questions
            try:
                questions = question_generator.generate_questions(
                    chunk_text,
                    n=num_questions_per_chunk
                )
            except Exception as e:
                print(f"Error generating questions for chunk {chunk_id}: {e}")
                questions = []

            # Create a pair for each question
            for question in questions:
                pair = {
                    'query': question,
                    'passage': chunk_text,
                    'doc_id': row.get('doc_id', 'unknown'),
                    'chunk_id': chunk_id,
                    'section_id': row.get('section_id', ''),
                    'section_title': row.get('section_title', ''),
                    'page_start': row.get('page_start', 0),
                    'page_end': row.get('page_end', 0),
                    'token_count': row.get('token_count', 0),
                    'char_count': row.get('char_count', 0)
                }
                pairs_data.append(pair)

            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{num_chunks} chunks...")

        # Create DataFrame
        pairs_df = pd.DataFrame(pairs_data)

        print(f"Generated {len(pairs_df)} query-passage pairs from {num_chunks} chunks")
        print(f"Average questions per chunk: {len(pairs_df) / num_chunks:.2f}")

        return pairs_df


def train_val_split_pairs(
    pairs_df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42,
    group_by: str = 'chunk_id'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split query-passage pairs into train/val sets with grouped splitting.

    Grouped splitting ensures that all pairs from the same chunk go to the same
    split, preventing data leakage.

    Args:
        pairs_df: DataFrame with query-passage pairs.
        val_ratio: Ratio of data to use for validation (0.0-1.0).
        random_state: Random seed for reproducibility.
        group_by: Column to group by for splitting (default: 'chunk_id').

    Returns:
        Tuple of (train_df, val_df) DataFrames.
    """

    if group_by not in pairs_df.columns:
        # Fall back to simple random split if group column doesn't exist
        print(f"Warning: {group_by} column not found, using simple random split")
        train_df, val_df = train_test_split(
            pairs_df,
            test_size=val_ratio,
            random_state=random_state
        )
        return train_df, val_df

    # Get unique groups (chunk_ids)
    unique_groups = pairs_df[group_by].unique()
    num_groups = len(unique_groups)

    # Split groups (not individual pairs)
    train_groups, val_groups = train_test_split(
        unique_groups,
        test_size=val_ratio,
        random_state=random_state
    )

    # Split pairs based on group membership
    train_df = pairs_df[pairs_df[group_by].isin(train_groups)].copy()
    val_df = pairs_df[pairs_df[group_by].isin(val_groups)].copy()

    print(f"Train: {len(train_df)} pairs from {len(train_groups)} chunks")
    print(f"Val: {len(val_df)} pairs from {len(val_groups)} chunks")

    return train_df, val_df


def save_questions_dataset(
    pairs_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    base_name: str = "pdf_query_passage_pairs"
) -> dict:
    """
    Save query-passage pairs to timestamped parquet and CSV files.

    Args:
        pairs_df: Full DataFrame with all pairs.
        train_df: Training split DataFrame (optional).
        val_df: Validation split DataFrame (optional).
        base_name: Base name for output files.

    Returns:
        Dictionary with paths to saved files.
    """

    saved_paths = {}

    # Save full dataset to parquet
    parquet_path = timestamped_path("data/processed", base_name, "parquet")
    pairs_df.to_parquet(parquet_path, index=False)
    saved_paths['full_parquet'] = str(parquet_path)
    print(f"Saved full dataset to: {parquet_path}")

    # Save full dataset to CSV (for easy inspection)
    csv_path = timestamped_path("data/processed", base_name, "csv")
    pairs_df.to_csv(csv_path, index=False)
    saved_paths['full_csv'] = str(csv_path)
    print(f"Saved full dataset to: {csv_path}")

    # Save train split if provided
    if train_df is not None:
        train_csv_path = timestamped_path("data/processed", f"{base_name}_train", "csv")
        train_df.to_csv(train_csv_path, index=False)
        saved_paths['train_csv'] = str(train_csv_path)
        print(f"Saved train split to: {train_csv_path}")

    # Save val split if provided
    if val_df is not None:
        val_csv_path = timestamped_path("data/processed", f"{base_name}_val", "csv")
        val_df.to_csv(val_csv_path, index=False)
        saved_paths['val_csv'] = str(val_csv_path)
        print(f"Saved val split to: {val_csv_path}")

    return saved_paths

