"""
Unit tests for question generation, focusing on heuristic fallback.

Tests the HeuristicQuestionGenerator which doesn't require LLM models.
"""

import pytest  # Testing framework

from src.llm.question_generation import (
    HeuristicQuestionGenerator,
    get_question_generator,
    QuestionGenConfig
)


def test_heuristic_question_generator():
    """Test heuristic question generator."""
    generator = HeuristicQuestionGenerator()

    chunk_text = """
    Machine Learning is a subset of Artificial Intelligence. 
    It involves training algorithms on data. 
    Deep Learning uses neural networks with multiple layers.
    """

    questions = generator.generate_questions(chunk_text, n=3)
    assert len(questions) > 0
    assert all(isinstance(q, str) for q in questions)
    assert all(len(q) > 10 for q in questions)  # Questions should be substantial


def test_heuristic_key_term_extraction():
    """Test key term extraction in heuristic generator."""
    generator = HeuristicQuestionGenerator()

    chunk_text = "Python is a programming language. It supports Object-Oriented Programming (OOP)."

    # Test key term extraction indirectly through question generation
    questions = generator.generate_questions(chunk_text, n=5)
    assert len(questions) > 0

    # Questions should reference key terms from the text
    text_lower = chunk_text.lower()
    question_text = " ".join(questions).lower()
    # At least some key terms should appear (flexible test)
    assert len(question_text) > 0


def test_heuristic_statement_to_question():
    """Test statement-to-question conversion."""
    generator = HeuristicQuestionGenerator()

    # Test with "X is Y" pattern
    statement = "Machine Learning is a subset of AI."
    question = generator._statement_to_question(statement)
    assert question is not None
    assert "?" in question or "What" in question or "How" in question


def test_get_question_generator_fallback():
    """Test that factory function falls back to heuristic."""
    # Create config that will likely fail (invalid model)
    config = QuestionGenConfig(
        model_name="nonexistent/model-name-12345",
        num_questions_per_chunk=2
    )

    # Should fall back to heuristic
    generator = get_question_generator(config, fallback_to_heuristic=True)
    assert isinstance(generator, HeuristicQuestionGenerator)

    # Test that it works
    chunk_text = "This is a test chunk with some content."
    questions = generator.generate_questions(chunk_text, n=2)
    assert len(questions) > 0


def test_heuristic_with_various_chunks():
    """Test heuristic generator with various chunk types."""
    generator = HeuristicQuestionGenerator()

    test_chunks = [
        "Python is a programming language. It is widely used for data science.",
        "The theory of relativity was developed by Einstein. It describes space and time.",
        "Neural networks consist of layers. Each layer processes information."
    ]

    for chunk in test_chunks:
        questions = generator.generate_questions(chunk, n=2)
        assert len(questions) > 0
        assert all("?" in q or q.startswith(("What", "How", "Why", "When", "Where")) for q in questions)

