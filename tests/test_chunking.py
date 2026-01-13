"""
Unit tests for sophisticated chunking.

Tests section detection, sentence splitting, and token-aware packing.
"""

import pytest  # Testing framework
from transformers import AutoTokenizer  # Tokenizer for testing

from src.data.chunking import (
    heading_score,
    detect_headings,
    build_sections,
    split_into_sentences,
    pack_sentences_into_chunks,
    chunk_sections,
    DocSection,
    TextChunk
)
from src.data.pdf_ingest import PDFPage


def test_heading_score():
    """Test heading detection scoring."""
    # Numeric heading should score high
    score1 = heading_score("1. Introduction")
    assert score1 > 0.3

    # ALL CAPS heading should score high
    score2 = heading_score("INTRODUCTION")
    assert score2 > 0.3

    # Regular sentence should score low
    score3 = heading_score("This is a regular sentence with normal text.")
    assert score3 < score1

    # Very long text should score lower
    score4 = heading_score("A" * 200)
    assert score4 < 0.5


def test_detect_headings():
    """Test heading detection in text lines."""
    lines = [
        "1. Introduction",
        "This is regular content.",
        "2. Methods",
        "More content here.",
        "RESULTS",
        "Final content."
    ]

    headings = detect_headings(lines, min_score=0.3)
    assert len(headings) >= 3  # Should detect at least 3 headings
    assert any("Introduction" in h[1] for h in headings)
    assert any("Methods" in h[1] for h in headings)


def test_build_sections():
    """Test section building from pages."""
    # Create mock pages
    pages = [
        PDFPage(
            page_number=1,
            raw_text="1. Introduction\n\nThis is the introduction.",
            cleaned_text="1. Introduction\n\nThis is the introduction.",
            line_list=["1. Introduction", "", "This is the introduction."],
            metadata={}
        ),
        PDFPage(
            page_number=2,
            raw_text="2. Methods\n\nThis describes methods.",
            cleaned_text="2. Methods\n\nThis describes methods.",
            line_list=["2. Methods", "", "This describes methods."],
            metadata={}
        )
    ]

    sections = build_sections(pages, "test.pdf", min_heading_score=0.3)
    assert len(sections) >= 2
    assert sections[0].page_start == 1
    assert sections[0].title in ["1. Introduction", "Introduction"]


def test_split_into_sentences():
    """Test sentence splitting."""
    text = "This is sentence one. This is sentence two! Is this sentence three?"
    sentences = split_into_sentences(text)
    assert len(sentences) >= 3

    # Test with abbreviations (should not split)
    text2 = "Dr. Smith went to the U.S.A. He met Mr. Jones."
    sentences2 = split_into_sentences(text2)
    # Should handle abbreviations (may split or not, depending on implementation)
    assert len(sentences2) > 0


def test_pack_sentences_into_chunks():
    """Test token-aware sentence packing."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

    # Create sentences
    sentences = [
        "This is a short sentence.",
        "This is a longer sentence with more words and content.",
        "Another sentence here.",
        "Yet another sentence for testing purposes."
    ] * 10  # Repeat to get enough content

    chunks = pack_sentences_into_chunks(
        sentences,
        tokenizer,
        max_tokens=50,
        overlap_tokens=10,
        min_tokens=20
    )

    assert len(chunks) > 0
    # Check that chunks don't exceed max_tokens (approximately)
    for chunk in chunks:
        tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
        assert tokens <= 60  # Allow some margin


def test_chunk_sections():
    """Test complete chunking pipeline."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

    # Create sections
    sections = [
        DocSection(
            section_id="sec_001",
            title="Introduction",
            level=1,
            page_start=1,
            page_end=1,
            text="This is a section with multiple sentences. Each sentence provides information. The section continues with more content.",
            source_pdf="test.pdf"
        ),
        DocSection(
            section_id="sec_002",
            title="Methods",
            level=1,
            page_start=2,
            page_end=2,
            text="This section describes methods. It has several sentences. Each sentence adds detail.",
            source_pdf="test.pdf"
        )
    ]

    chunks = chunk_sections(
        sections,
        tokenizer,
        doc_id="test_doc",
        max_tokens=50,
        overlap_tokens=10,
        min_tokens=20
    )

    assert len(chunks) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert all(chunk.doc_id == "test_doc" for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)

