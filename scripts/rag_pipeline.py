"""
RAG Pipeline for Hellobooks AI

This module implements a Retrieval-Augmented Generation (RAG) system:
    User Question → Retrieve relevant document chunks → Generate answer from context

Embedding: Pure Python TF-IDF (no external ML libraries needed)
Retrieval: Cosine similarity on TF-IDF vectors
Generation: Synthesizes relevant chunks into a concise, focused answer
"""

import os
import json
import math
import re
from collections import Counter


# ── Stop words and tokenizer (must match create_embeddings.py) ──────

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


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: count / total for word, count in counts.items()}


def compute_tfidf_vector(tf: dict, idf: dict) -> dict[str, float]:
    return {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ── Main RAG class ──────────────────────────────────────────────────

class HellobooksRAG:
    """
    Retrieval-Augmented Generation system for accounting Q&A.

    Architecture:
        1. User asks a question
        2. Auto-sync: Check if knowledge base has been updated, rebuild if necessary
        3. Question is turned into a TF-IDF vector
        4. Cosine similarity finds relevant chunks from the knowledge base
        5. Retrieved chunks are combined into a clear, concise answer
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.kb_path = os.path.join(base_dir, "knowledge_base")
        self.store_dir = os.path.join(base_dir, "vector_store")
        self.store_file = os.path.join(self.store_dir, "store.json")

        # Automatically rebuild store if knowledge base changed
        self._check_for_updates()

        print("[System] Loading vector store...")
        with open(self.store_file, "r", encoding="utf-8") as f:
            store = json.load(f)

        self.chunks = store["chunks"]
        self.idf = store["idf"]
        self.tfidf_vectors = store["tfidf_vectors"]

        print(f"[System] Loaded {len(self.chunks)} document chunks.")
        print("[System] RAG system ready.")

    def _check_for_updates(self):
        """
        Check if any .md files in knowledge_base are newer than store.json.
        If so, rebuild the vector store automatically.
        """
        try:
            from scripts.create_embeddings import build_vector_store
            
            needs_rebuild = False
            
            if not os.path.exists(self.store_file):
                print("[System] Vector store not found. Building for the first time...")
                needs_rebuild = True
            else:
                store_mtime = os.path.getmtime(self.store_file)
                
                # Check timestamps of all markdown files
                for root, _, files in os.walk(self.kb_path):
                    for file in files:
                        if file.endswith(".md"):
                            file_path = os.path.join(root, file)
                            if os.path.getmtime(file_path) > store_mtime:
                                print(f"[System] Update detected in {file}. Rebuilding vector store...")
                                needs_rebuild = True
                                break
                    if needs_rebuild:
                        break
            
            if needs_rebuild:
                build_vector_store()
                
        except Exception as e:
            print(f"[!] Warning: Auto-sync failed: {e}")
            if not os.path.exists(self.store_file):
                raise RuntimeError("Critical: Knowledge base not found and auto-sync failed.")

    def _retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the top-k most relevant document chunks for a question.
        Uses TF-IDF cosine similarity.
        """
        tokens = tokenize(question)
        tf = compute_tf(tokens)
        query_vec = compute_tfidf_vector(tf, self.idf)

        scored = []
        for i, chunk_vec in enumerate(self.tfidf_vectors):
            score = cosine_similarity(query_vec, chunk_vec)
            if score > 0.0:
                scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for score, idx in top:
            results.append({
                "text": self.chunks[idx]["text"],
                "score": score,
                "source": self.chunks[idx].get("source", "unknown"),
                "topic": self.chunks[idx].get("topic", "unknown"),
            })
        return results

    def _generate(self, question: str, context_chunks: list[dict]) -> str:
        """
        Generate a concise answer from retrieved context chunks.
        Extracts only the relevant point text (not the topic prefix)
        and formats them as a clean numbered list.
        """
        if not context_chunks:
            return (
                "I could not find relevant information to answer your question. "
                "Please try asking about bookkeeping, invoices, profit and loss, "
                "balance sheets, or cash flow."
            )

        # Determine the primary topic from the top result
        primary_topic = context_chunks[0]["topic"]

        # Extract the actual content (after the topic prefix line)
        answer_points = []
        for chunk in context_chunks:
            text = chunk["text"]
            # The chunk format is: "Topic: ...\nActual content line"
            lines = text.strip().split("\n")
            if len(lines) > 1:
                # Get everything after the topic prefix line
                content = "\n".join(lines[1:]).strip()
            else:
                content = text.strip()

            if content and content not in answer_points:
                answer_points.append(content)

        # Build the answer
        header = f"Here is what I found about {primary_topic}:\n\n"

        # Format as numbered list
        formatted_points = []
        for i, point in enumerate(answer_points, 1):
            formatted_points.append(f"{i}. {point}")

        answer = header + "\n".join(formatted_points)

        # Add source
        sources = set()
        for chunk in context_chunks:
            sources.add(chunk["topic"])
        source_line = ", ".join(sorted(sources))
        answer += f"\n\n(Source: {source_line})"

        return answer

    def answer_question(self, question: str) -> str:
        """
        Main RAG pipeline entry point.
          User Question → Retrieve relevant docs → Generate answer
        """
        retrieved = self._retrieve(question, top_k=5)
        answer = self._generate(question, retrieved)
        return answer
