"""
Retrieval-Augmented Generation (RAG) Pipeline

Responsible for servicing user queries by searching the BM25 vector store and 
returning highly relevant, human-readable answers. Supports auto-rebuilding 
if the knowledge base is edited during runtime.
"""

import os
import json
import re
import logging
from typing import List, Dict, Tuple

from scripts.text_processing import tokenize_text

# Define boundaries to protect against Resource Exhaustion (DoS)
MAX_QUERY_LENGTH = 500

class HellobooksRAG:
    """
    Core RAG orchestrator that dynamically updates the index,
    scores queries using BM25, and generates formatted responses.
    """

    # BM25 Tuning Parameters
    BM25_K1 = 1.5
    BM25_B = 0.75

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.kb_path = os.path.join(base_dir, "knowledge_base")
        self.store_dir = os.path.join(base_dir, "vector_store")
        self.store_file = os.path.join(self.store_dir, "store.json")

        # Initial load and auto-sync
        self._check_for_updates()
        self._load_store()

    def _load_store(self):
        """Safely loads the pre-computed BM25 indices from the disk."""
        logging.info("Loading vector store into memory...")
        try:
            # Defensive check to ensure we aren't loading a massive malicious file
            if os.path.exists(self.store_file) and os.path.getsize(self.store_file) > 50 * 1024 * 1024:
                raise ValueError("Vector store exceeds maximum safe size (50MB).")

            with open(self.store_file, "r", encoding="utf-8") as f:
                store = json.load(f)

            # Defensive typing and fallback defaults for loaded data
            self.chunks = store.get("chunks", [])
            self.idf = store.get("idf", {})
            self.doc_term_counts = store.get("doc_term_counts", [])
            self.doc_lengths = store.get("doc_lengths", [])
            self.avgdl = store.get("avgdl", 1.0)
            
            # Validate core structures exist to prevent downstream crashes
            if not isinstance(self.chunks, list) or not isinstance(self.idf, dict):
                 raise ValueError("Corrupted vector store format.")

        except FileNotFoundError:
            raise RuntimeError("Vector store not found. Ensure indexing completes before querying.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse vector store JSON: {e}")
            raise RuntimeError("Vector store is corrupted. Please rebuild the index.")


    def _check_for_updates(self) -> bool:
        """
        Dynamically compares timestamps of markdown files against the index.
        Rebuilds the index if new or modified files are detected.
        
        Returns:
            bool: True if an update and rebuild occurred, False otherwise.
        """
        try:
            from scripts.create_embeddings import build_vector_store
            
            if not os.path.exists(self.store_file):
                print("[System] Vector store missing. Building initial index...")
                build_vector_store()
                return True
                
            store_mtime = os.path.getmtime(self.store_file)
            
            # Scan knowledge base for newer files using safe traversal
            needs_rebuild = False
            for root, _, files in os.walk(self.kb_path):
                # Hardened against traversal by resolving realpath
                safe_root = os.path.realpath(root)
                if not safe_root.startswith(os.path.realpath(self.kb_path)):
                    continue
                    
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(safe_root, file)
                        if os.path.getsize(file_path) > 5 * 1024 * 1024:
                            continue # Skip excessively large files
                            
                        if os.path.getmtime(file_path) > store_mtime:
                            logging.info(f"Outdated index detected due to changes in {file}.")
                            needs_rebuild = True
                            break
                if needs_rebuild:
                    break
            
            if needs_rebuild:
                logging.info("Commencing live rebuilt of vector store...")
                build_vector_store()
                return True
            
            return False
                
        except Exception as e:
            logging.error(f"Auto-sync mechanism failed securely: {e}")
            if not os.path.exists(self.store_file):
                raise RuntimeError("Critical Error: Knowledge base missing.")
            return False

    def _calculate_bm25_score(self, query_tokens: List[str], chunk_idx: int, term_name: str) -> float:
        """
        Calculates the relevance score for a single chunk against the query
        using the BM25 algorithm and custom term-boosting.
        """
        if chunk_idx >= len(self.doc_lengths) or chunk_idx >= len(self.doc_term_counts):
            return 0.0

        doc_len = self.doc_lengths[chunk_idx]
        doc_counts = self.doc_term_counts[chunk_idx]
        
        term_name_words = re.findall(r"[a-z0-9]+", term_name.lower())
        term_score_boost = 1.0
        
        score = 0.0
        for term in query_tokens:
            # Add intelligence boost: If the user is specifically querying the defined term, heavily prioritize it
            if term in term_name_words:
                term_score_boost += 10.0
            elif term in term_name.lower():
                term_score_boost += 2.0
                
            if term in doc_counts and term in self.idf:
                tf = doc_counts[term]
                idf = self.idf[term]
                
                # Standard BM25 scoring formula
                numerator = tf * (self.BM25_K1 + 1)
                denominator = tf + self.BM25_K1 * (1 - self.BM25_B + self.BM25_B * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)
        
        # Apply the final term relevance multiplier
        return score * term_score_boost

    def _retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Tokenizes the user query and searches the index for the most contextually relevant chunks.
        """
        tokens = tokenize_text(question)
        if not tokens:
            return []

        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            term_name = chunk.get("term", "")
            score = self._calculate_bm25_score(tokens, i, term_name)
            
            # Minimum required threshold to filter out vague partial matches
            if score > 0.1:
                scored_chunks.append((score, i))

        # Sort descending by highest relevance score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_indices = scored_chunks[:top_k]

        results = []
        for score, idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "score": score,
                "source": self.chunks[idx].get("source", "unknown"),
                "topic": self.chunks[idx].get("topic", "unknown"),
            })
        return results

    def _generate(self, context_chunks: List[Dict[str, str]]) -> str:
        """
        Synthesizes the retrieved chunks into a clean, numbered, human-readable answer.
        """
        if not context_chunks:
            return (
                "I could not find relevant information to answer your question. "
                "Please check the existing knowledge base, or add this information to a markdown file."
            )

        # Utilize the most relevant chunk's topic to set the context
        primary_topic = context_chunks[0]["topic"]

        answer_points = []
        for chunk in context_chunks:
            text = chunk["text"]
            lines = text.strip().split("\n")
            
            # Omit the "Topic: XYZ" context prefix when presenting to the user
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else text.strip()

            # Prevent duplicate information
            if content and content not in answer_points:
                answer_points.append(content)

        header = f"Here is what I found about {primary_topic}:\n\n"
        formatted_points = [f"{i}. {point}" for i, point in enumerate(answer_points, 1)]
        
        sources = sorted(list({chunk["topic"] for chunk in context_chunks}))
        footer = f"\n\n(Source: {', '.join(sources)})"

        return header + "\n".join(formatted_points) + footer

    def answer_question(self, question: str) -> str:
        """
        The primary operational entry point.
        Checks for live updates -> Retrieves Data -> Formats Response.
        """
        # Strict Input Validation to prevent abuse
        if not isinstance(question, str):
            logging.warning("Non-string input rejected.")
            return "Invalid question format."
            
        if len(question) > MAX_QUERY_LENGTH:
            logging.warning(f"Query length {len(question)} exceeded max limit.")
            return f"Query too long. Please restrict to {MAX_QUERY_LENGTH} characters."

        if self._check_for_updates():
            self._load_store()
            
        retrieved_context = self._retrieve(question, top_k=5)
        return self._generate(retrieved_context)

