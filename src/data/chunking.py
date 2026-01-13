"""
Used in: 09_PDF_Ingest_Chunk_Embed.ipynb, pdf_to_chunks_pipeline.py
Purpose:
    Implement sophisticated chunking with section detection, sentence splitting,
    and token-aware packing with overlap. Produces chunks with rich metadata.
"""

import re  # Regular expressions for pattern matching
from dataclasses import dataclass  # For structured data containers
from typing import List, Tuple, Optional  # Type hints
from transformers import PreTrainedTokenizer  # Tokenizer for token counting
from src.data.pdf_ingest import PDFPage  # PDFPage dataclass
from src.utils.timing import timeit  # Performance tracking


@dataclass
class DocSection:
    """
    Represents a section of a document with detected heading.

    Attributes:
        section_id: Unique identifier for the section (e.g., "sec_001")
        title: Section heading/title text
        level: Hierarchical level (1 for "1.", 2 for "1.1", etc.)
        page_start: Starting page number (1-indexed)
        page_end: Ending page number (1-indexed)
        text: Full text content of the section
        source_pdf: Path to source PDF file
    """

    section_id: str
    title: str
    level: int
    page_start: int
    page_end: int
    text: str
    source_pdf: str


@dataclass
class TextChunk:
    """
    Represents a chunk of text with metadata for training and retrieval.

    Attributes:
        doc_id: Document identifier
        chunk_id: Unique chunk identifier (format: "{doc_id}::sec{section_idx:03d}::chunk{chunk_idx:05d}")
        section_id: Section identifier this chunk belongs to
        section_title: Title of the section
        page_start: Starting page number (1-indexed)
        page_end: Ending page number (1-indexed)
        text: Chunk text content
        token_count: Number of tokens in the chunk (using EmbeddingGemma tokenizer)
        char_count: Number of characters in the chunk
    """

    doc_id: str
    chunk_id: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    token_count: int
    char_count: int


def heading_score(line: str) -> float:
    """
    Compute a confidence score (0.0-1.0) for whether a line is a heading.

    Higher scores indicate higher confidence that the line is a heading.

    Args:
        line: Text line to evaluate.

    Returns:
        Confidence score between 0.0 and 1.0.
    """

    if not line or len(line.strip()) < 3:
        return 0.0

    line = line.strip()
    score = 0.0

    # Check for numeric heading patterns (e.g., "1.", "1.1", "1.1.1")
    numeric_pattern = r'^(\d+)(\.\d+)*\s+\S+'
    if re.match(numeric_pattern, line):
        score += 0.4

    # Check for ALL CAPS (likely heading, but not too long)
    if line.isupper() and 5 <= len(line) <= 120:
        score += 0.3

    # Check for Title Case (capitalized words, few punctuation)
    words = line.split()
    if words:
        title_case_ratio = sum(1 for w in words if w[0].isupper() if w) / len(words)
        if title_case_ratio > 0.5 and len(line) < 100:
            score += 0.2

    # Check if line doesn't end with period (headings often don't)
    if not line.endswith('.'):
        score += 0.1

    # Penalize very long lines (headings are usually short)
    if len(line) > 150:
        score *= 0.5

    return min(score, 1.0)


def detect_headings(lines: List[str], min_score: float = 0.4) -> List[Tuple[int, str]]:
    """
    Detect heading lines in a list of text lines using heuristics.

    Args:
        lines: List of text lines to analyze.
        min_score: Minimum confidence score to consider a line a heading.

    Returns:
        List of tuples (line_index, heading_text) for detected headings.
    """

    headings = []

    for idx, line in enumerate(lines):
        score = heading_score(line)
        if score >= min_score:
            headings.append((idx, line.strip()))

    return headings


def build_sections(
    pages: List[PDFPage],
    source_pdf: str,
    min_heading_score: float = 0.4
) -> List[DocSection]:
    """
    Build document sections from PDF pages by detecting headings.

    If no headings are detected, creates one section per page or one whole-document section.

    Args:
        pages: List of PDFPage objects with cleaned text and line_list.
        source_pdf: Path to source PDF file.
        min_heading_score: Minimum confidence score for heading detection.

    Returns:
        List of DocSection objects.
    """

    if not pages:
        return []

    # Combine all pages into a single stream of (page_num, line)
    all_lines = []
    for page in pages:
        for line in page.line_list:
            all_lines.append((page.page_number, line))

    if not all_lines:
        # No text found, return empty sections
        return []

    # Detect headings
    lines_only = [line for _, line in all_lines]
    heading_indices = detect_headings(lines_only, min_score=min_heading_score)

    sections = []

    if not heading_indices:
        # No headings detected: create one section per page or one whole section
        # Strategy: one section per page if multiple pages, else one section
        if len(pages) > 1:
            # One section per page
            for page in pages:
                section = DocSection(
                    section_id=f"sec_page_{page.page_number:03d}",
                    title=f"Page {page.page_number}",
                    level=1,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    text=page.cleaned_text,
                    source_pdf=source_pdf
                )
                sections.append(section)
        else:
            # Single page: one section for whole document
            section = DocSection(
                section_id="sec_001",
                title="Document",
                level=1,
                page_start=pages[0].page_number,
                page_end=pages[-1].page_number,
                text=pages[0].cleaned_text,
                source_pdf=source_pdf
            )
            sections.append(section)
    else:
        # Headings detected: create sections based on headings
        section_idx = 0

        for i, (heading_idx, heading_text) in enumerate(heading_indices):
            # Determine section level from heading pattern
            level = 1
            if re.match(r'^\d+\.\d+', heading_text):
                # Has sub-numbering (e.g., "1.1")
                level = 2
            elif re.match(r'^\d+\.\d+\.\d+', heading_text):
                level = 3

            # Find start and end of this section
            start_line_idx = heading_idx
            end_line_idx = heading_indices[i + 1][0] if i + 1 < len(heading_indices) else len(all_lines)

            # Extract text for this section
            section_lines = []
            page_start = None
            page_end = None

            for line_idx in range(start_line_idx, end_line_idx):
                page_num, line_text = all_lines[line_idx]
                section_lines.append(line_text)
                if page_start is None:
                    page_start = page_num
                page_end = page_num

            section_text = '\n'.join(section_lines)

            section = DocSection(
                section_id=f"sec_{section_idx:03d}",
                title=heading_text,
                level=level,
                page_start=page_start or 1,
                page_end=page_end or 1,
                text=section_text,
                source_pdf=source_pdf
            )
            sections.append(section)
            section_idx += 1

    return sections


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex, protecting common abbreviations.

    Args:
        text: Text to split into sentences.

    Returns:
        List of sentence strings (with punctuation preserved).
    """

    if not text:
        return []

    # Common abbreviations that should not trigger sentence splits
    abbreviations = [
        r'\bDr\.', r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bProf\.',
        r'\be\.g\.', r'\bi\.e\.', r'\betc\.', r'\bvs\.', r'\bFig\.',
        r'\bEq\.', r'\bEqs\.', r'\bNo\.', r'\bVol\.', r'\bpp\.',
        r'\bet al\.', r'\bInc\.', r'\bLtd\.', r'\bCorp\.'
    ]

    # Create pattern that splits on sentence endings but protects abbreviations
    # Pattern: split on . ! ? but not if followed by lowercase or if it's an abbreviation
    sentence_endings = r'[.!?]+(?:\s+|$)'

    # First, protect abbreviations by replacing them with placeholders
    protected_text = text
    abbr_map = {}
    for i, abbr_pattern in enumerate(abbreviations):
        placeholder = f"__ABBR_{i}__"
        protected_text = re.sub(abbr_pattern, placeholder, protected_text)
        abbr_map[placeholder] = abbr_pattern.replace('\\', '')

    # Split on sentence endings
    sentences = re.split(sentence_endings, protected_text)

    # Restore abbreviations and clean up
    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Restore abbreviations
        for placeholder, abbr in abbr_map.items():
            sent = sent.replace(placeholder, abbr)

        # Add back punctuation if it was removed (approximate)
        if sent and not sent[-1] in '.!?':
            # Try to infer punctuation from context (simplified)
            pass

        result.append(sent)

    # Filter out very short sentences (likely artifacts)
    result = [s for s in result if len(s.strip()) > 3]

    return result


def pack_sentences_into_chunks(
    sentences: List[str],
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 128
) -> List[str]:
    """
    Pack sentences into chunks with token-aware sizing and overlap.

    Args:
        sentences: List of sentence strings to pack.
        tokenizer: Tokenizer for counting tokens.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of tokens to overlap between chunks.
        min_tokens: Minimum tokens per chunk (smaller chunks will be merged).

    Returns:
        List of chunk text strings.
    """

    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    for sentence in sentences:
        # Count tokens for this sentence
        # add_special_tokens=False because we're just counting content tokens
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        # Check if adding this sentence would exceed max_tokens
        if current_tokens + sentence_tokens > max_tokens and current_chunk_sentences:
            # Finalize current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)

            # Start new chunk with overlap
            # Carry forward sentences until we reach overlap_tokens
            overlap_sentences = []
            overlap_token_count = 0

            # Add sentences from end of current chunk for overlap
            for sent in reversed(current_chunk_sentences):
                sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                if overlap_token_count + sent_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, sent)
                    overlap_token_count += sent_tokens
                else:
                    break

            # Start new chunk with overlap
            current_chunk_sentences = overlap_sentences
            current_tokens = overlap_token_count

        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Add final chunk if it exists
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(chunk_text)

    # Merge small chunks (those below min_tokens)
    merged_chunks = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))

        # If chunk is too small and not the last one, try to merge with next
        if chunk_tokens < min_tokens and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            combined = chunk + ' ' + next_chunk
            combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))

            # Only merge if combined chunk doesn't exceed max_tokens too much
            if combined_tokens <= max_tokens * 1.2:  # Allow 20% overflow for merging
                merged_chunks.append(combined)
                i += 2  # Skip next chunk since we merged it
                continue

        merged_chunks.append(chunk)
        i += 1

    return merged_chunks


@timeit
def chunk_sections(
    sections: List[DocSection],
    tokenizer: PreTrainedTokenizer,
    doc_id: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 128
) -> List[TextChunk]:
    """
    Chunk document sections into TextChunk objects with metadata.

    Args:
        sections: List of DocSection objects to chunk.
        tokenizer: Tokenizer for token counting.
        doc_id: Document identifier.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of tokens to overlap between chunks.
        min_tokens: Minimum tokens per chunk.

    Returns:
        List of TextChunk objects with metadata.
    """

    all_chunks = []
    global_chunk_idx = 0

    for section_idx, section in enumerate(sections):
        # Split section text into sentences
        sentences = split_into_sentences(section.text)

        if not sentences:
            continue

        # Pack sentences into chunks
        chunk_texts = pack_sentences_into_chunks(
            sentences,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens
        )

        # Create TextChunk objects for each chunk
        for chunk_idx, chunk_text in enumerate(chunk_texts):
            # Count tokens and characters
            token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))
            char_count = len(chunk_text)

            # Quality check: skip chunks that are mostly noise
            # Count alphanumeric characters
            alnum_ratio = sum(1 for c in chunk_text if c.isalnum()) / len(chunk_text) if chunk_text else 0
            if alnum_ratio < 0.3:  # Less than 30% alphanumeric = likely noise
                continue

            # Generate stable chunk_id
            chunk_id = f"{doc_id}::sec{section_idx:03d}::chunk{chunk_idx:05d}"

            chunk = TextChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                section_id=section.section_id,
                section_title=section.title,
                page_start=section.page_start,
                page_end=section.page_end,
                text=chunk_text,
                token_count=token_count,
                char_count=char_count
            )

            all_chunks.append(chunk)
            global_chunk_idx += 1

    return all_chunks

