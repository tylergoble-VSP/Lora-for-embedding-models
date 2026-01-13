"""
Used in: 09_PDF_Ingest_Chunk_Embed.ipynb, pdf_to_chunks_pipeline.py
Purpose:
    Clean and normalize text extracted from PDFs, including whitespace normalization,
    hyphenation fixing, and header/footer removal.
"""

import re  # Regular expressions for text pattern matching
from typing import List, TYPE_CHECKING  # Type hints
from collections import Counter  # For counting line frequencies
from src.utils.timing import timeit  # Performance tracking

if TYPE_CHECKING:
    from src.data.pdf_ingest import PDFPage


@timeit
def normalize_text(text: str) -> str:
    """
    Normalize text by cleaning whitespace, fixing hyphenation, and removing artifacts.

    This function:
    1. Removes null characters and other control characters
    2. Normalizes newlines (converts all to single newline)
    3. Collapses repeated whitespace
    4. Fixes hyphenation across line breaks (e.g., "exam-\nple" -> "example")
    5. Removes common ligatures and special characters

    Args:
        text: Raw text string to normalize.

    Returns:
        Normalized text string.
    """

    if not text:
        return ""

    # Remove null characters and other problematic control characters
    # Keep newlines, tabs, and spaces for now
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)

    # Normalize newlines: convert all newline variants to single \n
    text = re.sub(r'\r\n|\r', '\n', text)

    # Fix hyphenation across line breaks
    # Pattern: word ending with hyphen, followed by newline, followed by continuation
    # Example: "exam-\nple" -> "example"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Remove common ligatures and special characters
    # Replace common ligatures with their ASCII equivalents
    ligature_map = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        '–': '-',  # en dash
        '—': '-',  # em dash
        '"': '"',  # left double quote
        '"': '"',  # right double quote
        ''': "'",  # left single quote
        ''': "'",  # right single quote
    }
    for lig, replacement in ligature_map.items():
        text = text.replace(lig, replacement)

    # Collapse repeated whitespace (but preserve single newlines)
    # First, collapse multiple spaces/tabs into single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Then, collapse multiple newlines into double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Finally, remove spaces at start/end of lines
    text = re.sub(r' +(\n)', r'\1', text)  # trailing spaces
    text = re.sub(r'(\n) +', r'\1', text)  # leading spaces

    return text.strip()


def split_lines(text: str) -> List[str]:
    """
    Split text into individual lines, strip whitespace, and drop empty lines.

    Args:
        text: Text string to split.

    Returns:
        List of non-empty lines (stripped of leading/trailing whitespace).
    """

    if not text:
        return []

    # Split by newline
    lines = text.split('\n')

    # Strip each line and filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]

    return lines


@timeit
def strip_repeated_headers_footers(
    pages: List['PDFPage'],
    top_k_lines: int = 3,
    bottom_k_lines: int = 3,
    freq_threshold: float = 0.6
) -> List['PDFPage']:
    """
    Remove repeated headers and footers from PDF pages.

    Headers/footers are identified as lines that appear on a high percentage
    of pages (>= freq_threshold). These are removed from each page.

    Args:
        pages: List of PDFPage objects (must have line_list populated).
        top_k_lines: Number of lines from top of each page to check for headers.
        bottom_k_lines: Number of lines from bottom of each page to check for footers.
        freq_threshold: Minimum frequency (0.0-1.0) for a line to be considered header/footer.
                       Default 0.6 means line must appear on >=60% of pages.

    Returns:
        List of PDFPage objects with headers/footers removed from line_list and cleaned_text updated.
    """

    # Import here to avoid circular dependency
    from src.data.pdf_ingest import PDFPage

    if not pages:
        return pages

    # First, ensure all pages have line_list populated
    for page in pages:
        if not page.line_list:
            page.line_list = split_lines(page.cleaned_text)

    # Collect candidate header/footer lines
    header_candidates = []
    footer_candidates = []

    for page in pages:
        lines = page.line_list
        if len(lines) >= top_k_lines:
            # Get top k lines as header candidates
            header_candidates.extend(lines[:top_k_lines])
        if len(lines) >= bottom_k_lines:
            # Get bottom k lines as footer candidates
            footer_candidates.extend(lines[-bottom_k_lines:])

    # Count frequency of each candidate line
    header_counts = Counter(header_candidates)
    footer_counts = Counter(footer_candidates)

    num_pages = len(pages)
    min_occurrences = int(freq_threshold * num_pages)

    # Identify lines that appear frequently enough to be headers/footers
    header_lines = {
        line for line, count in header_counts.items()
        if count >= min_occurrences
    }
    footer_lines = {
        line for line, count in footer_counts.items()
        if count >= min_occurrences
    }

    # Remove identified headers/footers from each page
    for page in pages:
        lines = page.line_list.copy()
        original_len = len(lines)

        # Remove header lines from top
        while lines and lines[0] in header_lines:
            lines.pop(0)

        # Remove footer lines from bottom
        while lines and lines[-1] in footer_lines:
            lines.pop()

        # Update page with filtered lines
        page.line_list = lines
        page.cleaned_text = '\n'.join(lines)

    return pages


@timeit
def load_and_clean_pdf(pdf_path: str) -> List['PDFPage']:
    """
    Complete pipeline: extract PDF pages and clean them.

    This function orchestrates:
    1. PDF text extraction (using extract_pdf_pages)
    2. Text normalization (using normalize_text)
    3. Line splitting (using split_lines)
    4. Header/footer removal (using strip_repeated_headers_footers)

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PDFPage objects with cleaned text and line lists.
    """

    from src.data.pdf_ingest import extract_pdf_pages

    # Step 1: Extract raw text from PDF
    pages = extract_pdf_pages(pdf_path)

    # Step 2: Normalize and clean each page
    for page in pages:
        # Normalize the raw text
        page.cleaned_text = normalize_text(page.raw_text)

        # Split into lines
        page.line_list = split_lines(page.cleaned_text)

    # Step 3: Remove repeated headers and footers
    pages = strip_repeated_headers_footers(pages)

    return pages

