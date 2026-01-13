"""
Used in: 10_Generate_Questions_Build_Dataset.ipynb, question_generation_dataset.py
Purpose:
    Generate questions from text chunks using HuggingFace text-generation models
    or deterministic heuristics as a fallback.
"""

import json  # JSON parsing for LLM responses
import re  # Regular expressions for text processing
from dataclasses import dataclass  # For structured configuration
from typing import List, Optional, Protocol  # Type hints
from abc import ABC, abstractmethod  # Abstract base classes

import torch  # PyTorch for model operations
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace model loading

from src.utils.timing import timeit  # Performance tracking


@dataclass
class QuestionGenConfig:
    """
    Configuration for question generation.

    Attributes:
        model_name: HuggingFace model identifier for text generation.
        device: Device to run on ('cuda', 'cpu', or None for auto-detection).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more creative).
        top_p: Nucleus sampling parameter.
        num_questions_per_chunk: Number of questions to generate per chunk.
        prompt_template: Template for question generation prompt.
    """

    model_name: str = "microsoft/DialoGPT-medium"  # Default lightweight model
    device: Optional[str] = None
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    num_questions_per_chunk: int = 3
    prompt_template: str = """Given the following passage, generate {num_questions} questions that can be answered from this passage. Make the questions clear and suitable for an undergraduate student.

Passage:
{chunk_text}

Questions (JSON array format, one question per element):"""


class BaseQuestionGenerator(ABC):
    """
    Abstract base class for question generators.

    All question generators must implement the generate_questions method.
    """

    @abstractmethod
    def generate_questions(self, chunk_text: str, n: int) -> List[str]:
        """
        Generate n questions from a text chunk.

        Args:
            chunk_text: Text chunk to generate questions from.
            n: Number of questions to generate.

        Returns:
            List of question strings.
        """
        pass


class HFQuestionGenerator(BaseQuestionGenerator):
    """
    Question generator using HuggingFace text-generation models.

    Uses a prompt template to generate questions, with robust JSON parsing
    and fallback handling.
    """

    def __init__(self, config: QuestionGenConfig):
        """
        Initialize the HuggingFace question generator.

        Args:
            config: QuestionGenConfig with model and generation parameters.
        """
        self.config = config

        # Auto-detect device if not specified
        if config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        print(f"Loading question generation model: {config.model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Set pad_token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            print(f"Failed to load model {config.model_name}: {e}")
            raise

    @timeit
    def generate_questions(self, chunk_text: str, n: int) -> List[str]:
        """
        Generate questions from a chunk using the HuggingFace model.

        Args:
            chunk_text: Text chunk to generate questions from.
            n: Number of questions to generate.

        Returns:
            List of question strings (may be fewer than n if parsing fails).
        """
        # Format prompt
        prompt = self.config.prompt_template.format(
            num_questions=n,
            chunk_text=chunk_text[:2000]  # Limit chunk length in prompt
        )

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate questions
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract questions from generated text
        questions = self._parse_questions(generated_text, prompt, n)

        return questions

    def _parse_questions(self, generated_text: str, prompt: str, n: int) -> List[str]:
        """
        Parse questions from generated text with multiple fallback strategies.

        Args:
            generated_text: Full generated text including prompt.
            prompt: Original prompt (to remove from generated text).
            n: Expected number of questions.

        Returns:
            List of parsed question strings.
        """
        # Remove prompt from generated text
        if prompt in generated_text:
            generated_text = generated_text.split(prompt, 1)[1].strip()

        questions = []

        # Strategy 1: Try to parse as JSON array
        try:
            # Look for JSON array pattern
            json_match = re.search(r'\[.*?\]', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    questions = [str(q).strip() for q in parsed if q]
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 2: Extract numbered questions (e.g., "1. What is...")
        if not questions:
            numbered_pattern = r'(?:^|\n)\s*(?:\d+[\.\)]|\-|\*)\s*(.+?)(?=\n\s*(?:\d+[\.\)]|\-|\*)|\n\n|$)'
            matches = re.findall(numbered_pattern, generated_text, re.MULTILINE)
            questions = [m.strip() for m in matches if m.strip()]

        # Strategy 3: Extract question mark patterns
        if not questions:
            question_pattern = r'([^.!?]*\?)'
            matches = re.findall(question_pattern, generated_text)
            questions = [m.strip() for m in matches if m.strip() and len(m) > 10]

        # Strategy 4: Split by newlines and filter for questions
        if not questions:
            lines = generated_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('?' in line or line.startswith('What') or line.startswith('How') or
                           line.startswith('Why') or line.startswith('When') or line.startswith('Where')):
                    # Remove numbering/bullets
                    line = re.sub(r'^[\d\.\)\-\*]\s*', '', line)
                    if len(line) > 10:
                        questions.append(line)

        # Post-process: deduplicate and filter
        questions = self._post_process_questions(questions, n)

        return questions

    def _post_process_questions(self, questions: List[str], n: int) -> List[str]:
        """
        Post-process questions: deduplicate, filter generic ones, limit to n.

        Args:
            questions: List of raw question strings.
            n: Target number of questions.

        Returns:
            Filtered and deduplicated list of questions.
        """
        # Deduplicate (case-insensitive)
        seen = set()
        unique_questions = []
        for q in questions:
            q_lower = q.lower().strip()
            if q_lower not in seen and len(q_lower) > 10:
                seen.add(q_lower)
                unique_questions.append(q)

        # Filter out generic questions
        generic_patterns = [
            r'^what is the',
            r'^what are the',
            r'^can you',
            r'^please',
            r'^tell me',
        ]
        filtered = []
        for q in unique_questions:
            q_lower = q.lower().strip()
            is_generic = any(re.match(pattern, q_lower) for pattern in generic_patterns)
            if not is_generic:
                filtered.append(q)

        # Limit to n questions
        return filtered[:n]


class HeuristicQuestionGenerator(BaseQuestionGenerator):
    """
    Deterministic question generator using heuristics (no LLM required).

    Extracts key terms and generates template-based questions.
    """

    def __init__(self):
        """Initialize the heuristic question generator (no model loading needed)."""
        pass

    @timeit
    def generate_questions(self, chunk_text: str, n: int) -> List[str]:
        """
        Generate questions from a chunk using heuristics.

        Args:
            chunk_text: Text chunk to generate questions from.
            n: Number of questions to generate.

        Returns:
            List of question strings.
        """
        questions = []

        # Extract key terms (capitalized words, repeated phrases, definitions)
        key_terms = self._extract_key_terms(chunk_text)

        # Generate questions using templates
        templates = [
            "What is {term}?",
            "How does {term} work?",
            "What are the characteristics of {term}?",
            "Why is {term} important?",
            "What is the relationship between {term1} and {term2}?",
            "How is {term} used?",
            "What are examples of {term}?",
        ]

        # Use key terms to fill templates
        for i, term in enumerate(key_terms[:n * 2]):  # Get more terms than needed
            if i >= n:
                break

            # Try single-term templates
            if i < len(templates) - 1:
                template = templates[i % (len(templates) - 1)]
                question = template.format(term=term)
                questions.append(question)
            else:
                # Try two-term template
                if i + 1 < len(key_terms):
                    term2 = key_terms[i + 1]
                    question = templates[-1].format(term1=term, term2=term2)
                    questions.append(question)

        # If we don't have enough questions, generate from sentence starts
        if len(questions) < n:
            sentences = self._extract_sentences(chunk_text)
            for sent in sentences[:n - len(questions)]:
                # Convert statement to question
                question = self._statement_to_question(sent)
                if question:
                    questions.append(question)

        return questions[:n]

    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text (capitalized words, repeated phrases).

        Args:
            text: Text to extract terms from.

        Returns:
            List of key term strings.
        """
        terms = []

        # Extract capitalized words (likely proper nouns or important concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        # Count frequency
        from collections import Counter
        cap_counts = Counter(capitalized)
        # Get most frequent (excluding common words)
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
        for term, count in cap_counts.most_common(10):
            if term not in common_words and len(term) > 3:
                terms.append(term)

        # Extract phrases in quotes or parentheses (often definitions)
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        parenthetical = re.findall(r'\(([^)]+)\)', text)
        for phrase in quoted + parenthetical:
            if len(phrase) > 5 and len(phrase) < 50:
                terms.append(phrase.strip())

        # Extract "X is Y" or "X are Y" patterns (definitions)
        definition_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are)\s+([^.,;]+)'
        definitions = re.findall(definition_pattern, text)
        for defn in definitions:
            terms.append(defn[0])  # The defined term

        return list(set(terms))[:20]  # Deduplicate and limit

    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.

        Args:
            text: Text to extract sentences from.

        Returns:
            List of sentence strings.
        """
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _statement_to_question(self, statement: str) -> Optional[str]:
        """
        Convert a statement to a question (simple heuristic).

        Args:
            statement: Statement string.

        Returns:
            Question string, or None if conversion fails.
        """
        statement = statement.strip()
        if not statement or len(statement) < 10:
            return None

        # Try to convert "X is Y" to "What is X?"
        is_match = re.match(r'^([A-Z][^.!?]+?)\s+is\s+([^.!?]+)', statement)
        if is_match:
            subject = is_match.group(1).strip()
            return f"What is {subject}?"

        # Try to convert "X does Y" to "How does X do Y?"
        does_match = re.match(r'^([A-Z][^.!?]+?)\s+(?:does|do|did)\s+([^.!?]+)', statement)
        if does_match:
            subject = does_match.group(1).strip()
            return f"How does {subject} work?"

        # Generic: "Tell me about X"
        words = statement.split()
        if len(words) > 3:
            # Take first few words as subject
            subject = ' '.join(words[:3])
            return f"What is {subject}?"

        return None


def get_question_generator(
    config: Optional[QuestionGenConfig] = None,
    fallback_to_heuristic: bool = True
) -> BaseQuestionGenerator:
    """
    Factory function to get a question generator (tries HF, falls back to heuristic).

    Args:
        config: QuestionGenConfig for HuggingFace generator (None uses defaults).
        fallback_to_heuristic: If True, fall back to heuristic if HF fails.

    Returns:
        BaseQuestionGenerator instance.
    """
    if config is None:
        config = QuestionGenConfig()

    # Try HuggingFace generator first
    if fallback_to_heuristic:
        try:
            return HFQuestionGenerator(config)
        except Exception as e:
            print(f"HuggingFace question generator failed: {e}")
            print("Falling back to heuristic generator...")
            return HeuristicQuestionGenerator()
    else:
        return HFQuestionGenerator(config)

