"""
Used in: 09_PDF_Ingest_Chunk_Embed.ipynb, pdf_to_chunks_pipeline.py
Purpose:
    Extract text from PDF files using pdfplumber (primary) with pypdf fallback.
    Handles page-by-page extraction with robust error handling.
"""

import pdfplumber  # Primary PDF extraction library (layout-aware)
import pypdf  # Fallback PDF extraction library
from dataclasses import dataclass  # For structured data containers
from typing import List, Dict, Optional  # Type hints
from pathlib import Path  # Path handling
from src.utils.timing import timeit, TimeBlock  # Performance tracking


@dataclass
class PDFPage:
    """
    Represents a single page from a PDF document.

    Attributes:
        page_number: 1-indexed page number in the PDF
        raw_text: Raw text extracted from the page (before cleaning)
        cleaned_text: Text after normalization and cleaning
        line_list: List of individual lines from the page
        metadata: Optional dictionary with additional page info (width, height, etc.)
    """

    page_number: int
    raw_text: str
    cleaned_text: str
    line_list: List[str]
    metadata: Dict


@timeit
def extract_pdf_pages(pdf_path: str) -> List[PDFPage]:
    """
    Extract text from all pages of a PDF file.

    Tries pdfplumber first (better layout preservation), falls back to pypdf
    if pdfplumber fails or returns empty pages.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PDFPage objects, one per page.

    Raises:
        FileNotFoundError: If PDF file doesn't exist.
        ValueError: If all pages are empty (likely scanned PDF, needs OCR).
    """

    pdf_path_obj = Path(pdf_path)

    # Check if file exists
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = []
    pdfplumber_success = False

    # Try pdfplumber first (better for layout-aware extraction)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text with tolerance settings for better layout handling
                raw_text = page.extract_text(
                    x_tolerance=3,  # Horizontal tolerance for combining words
                    y_tolerance=3  # Vertical tolerance for combining lines
                )

                # If extract_text returns None, try without tolerance
                if raw_text is None:
                    raw_text = page.extract_text() or ""

                # Get page metadata if available
                page_metadata = {}
                if hasattr(page, 'width') and hasattr(page, 'height'):
                    page_metadata['width'] = page.width
                    page_metadata['height'] = page.height

                pages.append(PDFPage(
                    page_number=page_num,
                    raw_text=raw_text,
                    cleaned_text="",  # Will be filled by cleaning step
                    line_list=[],  # Will be filled by cleaning step
                    metadata=page_metadata
                ))

            # Check if we got meaningful text from at least some pages
            non_empty_count = sum(1 for p in pages if p.raw_text.strip())
            if non_empty_count > 0:
                pdfplumber_success = True

    except Exception as e:
        # pdfplumber failed, will try pypdf fallback
        print(f"pdfplumber extraction failed: {e}")
        print("Falling back to pypdf...")

    # Fallback to pypdf if pdfplumber failed or returned mostly empty pages
    if not pdfplumber_success or len(pages) == 0:
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(1, num_pages + 1):
                    page = pdf_reader.pages[page_num - 1]
                    raw_text = page.extract_text() or ""

                    pages.append(PDFPage(
                        page_number=page_num,
                        raw_text=raw_text,
                        cleaned_text="",
                        line_list=[],
                        metadata={}
                    ))

        except Exception as e:
            raise RuntimeError(f"Both pdfplumber and pypdf failed to extract text: {e}")

    # Validate that we have at least some non-empty pages
    non_empty_pages = [p for p in pages if p.raw_text.strip()]
    if len(non_empty_pages) == 0:
        raise ValueError(
            f"All pages in PDF are empty. This might be a scanned PDF requiring OCR. "
            f"File: {pdf_path}"
        )

    return pages

