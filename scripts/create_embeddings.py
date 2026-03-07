"""
Embedding Generation Script for Hellobooks AI

This script:
1. Loads all markdown documents from the knowledge_base/ folder
2. Splits them into fine-grained chunks (one per numbered point)
3. Computes TF-IDF vectors for each chunk (pure Python implementation)
4. Saves the vector store to disk as JSON

Usage:
    python scripts/create_embeddings.py
"""

import os
import json
import math
import re
from collections import Counter


# ── Stop words ──────────────────────────────────────────────────────

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


# ── Tokenizer ───────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric chars, remove stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


# ── TF-IDF computation ─────────────────────────────────────────────

def compute_tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: count / total for word, count in counts.items()}


def compute_idf(documents_tokens: list[list[str]]) -> dict[str, float]:
    n_docs = len(documents_tokens)
    df = Counter()
    for tokens in documents_tokens:
        for token in set(tokens):
            df[token] += 1
    return {word: math.log(n_docs / (1 + count)) for word, count in df.items()}


def compute_tfidf_vector(tf: dict, idf: dict) -> dict[str, float]:
    return {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}


# ── Document loading and chunking ──────────────────────────────────

def load_markdown_files(folder: str) -> list[dict]:
    """Read all .md files from a folder."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".md"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract topic name from filename
            topic = filename.replace(".md", "").replace("_", " ").title()
            docs.append({"text": content, "source": filepath, "topic": topic})
    return docs


def split_into_chunks(text: str, source: str, topic: str) -> list[dict]:
    """
    Split a document into fine-grained chunks.
    Each numbered point (e.g., 1. Name: Description) becomes its own chunk.
    The document title and description are prepended to each chunk
    for better semantic context during retrieval.
    """
    lines = text.strip().split("\n")
    chunks = []

    # Extract the document title and description
    doc_title = ""
    doc_description = ""
    content_lines = []

    # Better regex for detecting the start of a multi-point list (1., 2., etc.)
    # or just content lines.
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        if stripped.startswith("# ") and not doc_title:
            doc_title = stripped.lstrip("# ").strip()
        elif stripped.startswith("**Description**"):
            doc_description = stripped.replace("**Description**:", "").replace("**Description**", "").strip()
        else:
            # Check if it starts with a number or is general content
            content_lines.append(stripped)

    # Build a context prefix that helps TF-IDF identify the correct topic
    context_prefix = f"Topic: {doc_title}."
    if doc_description:
        context_prefix += f" {doc_description}"

    # Split each line (numbered point) into its own chunk
    for line in content_lines:
        # Detect numbered points like "1. Name: Description" or "20. Name"
        # and clean up formatting
        clean_line = re.sub(r"^\d+[\.\)]\s*", "", line)
        clean_line = clean_line.replace("**", "")

        # Combine the topic context with the specific point content
        chunk_text = f"{context_prefix}\n{clean_line}"

        chunks.append({
            "text": chunk_text,
            "source": source,
            "topic": topic
        })

    # Fallback if no specific points were extracted
    if not chunks:
        chunks.append({
            "text": f"{context_prefix}\n{text.strip()}",
            "source": source,
            "topic": topic
        })

    return chunks


def build_vector_store():
    """Reads documents, chunks them, generates TF-IDF vectors, and saves to JSON."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_base_path = os.path.join(base_dir, "knowledge_base")
    vector_store_path = os.path.join(base_dir, "vector_store")

    if not os.path.exists(knowledge_base_path):
        print(f"[!] Knowledge base folder not found: {knowledge_base_path}")
        return

    os.makedirs(vector_store_path, exist_ok=True)

    # Step 1: Load markdown files
    print("[*] Rebuilding vector store from knowledge_base/...")
    documents = load_markdown_files(knowledge_base_path)
    if not documents:
        print("[!] No markdown files found.")
        return

    # Step 2: Split into fine-grained chunks
    all_chunks = []
    for doc in documents:
        doc_chunks = split_into_chunks(doc["text"], doc["source"], doc["topic"])
        all_chunks.extend(doc_chunks)
    
    # Step 3: Tokenize
    all_tokens = [tokenize(chunk["text"]) for chunk in all_chunks]

    # Step 4: Compute IDF
    idf = compute_idf(all_tokens)

    # Step 5: Compute TF-IDF vectors
    tfidf_vectors = []
    for tokens in all_tokens:
        tf = compute_tf(tokens)
        vec = compute_tfidf_vector(tf, idf)
        tfidf_vectors.append(vec)

    # Step 6: Save
    store = {
        "chunks": all_chunks,
        "idf": idf,
        "tfidf_vectors": tfidf_vectors,
    }

    store_file = os.path.join(vector_store_path, "store.json")
    with open(store_file, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

    print(f"[*] Rebuild complete: {len(all_chunks)} chunks indexed in {store_file}")


if __name__ == "__main__":
    build_vector_store()


if __name__ == "__main__":
    build_vector_store()
