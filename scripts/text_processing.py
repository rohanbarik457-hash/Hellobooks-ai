"""
Common text processing utilities used across the RAG pipeline.
This ensures the tokenizer and stop words are consistent during both
indexing and retrieval (DRY principle).
"""
import re
from collections import Counter
from typing import List, Dict

# Complete list of common English stop words to filter out non-informative terms
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "with", "as", "at", "by", "from", "that", "this", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "can",
    "could", "not", "no", "but", "if", "so", "than", "too", "very",
    "just", "about", "also", "into", "over", "such", "its", "your",
    "our", "their", "we", "you", "he", "she", "they", "them", "his",
    "her", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "any", "these", "those", "what", "which", "who",
    "how", "when", "where", "why", "up", "out", "then", "here", "there",
}

def tokenize_text(text: str) -> List[str]:
    """
    Cleans and tokenizes text by removing non-alphanumeric characters
    and filtering out common stop words.

    Args:
        text (str): The raw text to process.

    Returns:
        List[str]: A list of clean, meaningful tokens.
    """
    # Extract only alphanumeric words, cast to lowercase for uniformity
    words = re.findall(r"[a-z0-9]+", text.lower())
    
    # Filter out empty strings, single-character words, and stop words
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]

def compute_term_frequencies(tokens: List[str]) -> Dict[str, int]:
    """
    Calculates the frequency of each term within a document.

    Args:
        tokens (List[str]): List of parsed tokens from a document.

    Returns:
        Dict[str, int]: A mapping of terms to their integer frequencies.
    """
    return dict(Counter(tokens))
