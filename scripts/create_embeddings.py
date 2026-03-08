"""
Vector Store Indexer

Responsible for loading markdown files from the knowledge base, chunking them
into discrete, searchable items, and pre-computing BM25 statistics for rapid
retrieval during queries.
"""

import os
import json
import math
import re
import logging
from collections import Counter
from typing import List, Dict, Any

from scripts.text_processing import tokenize_text, compute_term_frequencies

# Security constants
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB threshold to prevent memory exhaustion

class KnowledgeBaseIndexer:
    """
    Handles reading markdown documents, parsing them into chunks, and
    compiling the BM25 statistical vector store.
    """

    def __init__(self, knowledge_base_dir: str, vector_store_dir: str):
        self.kb_dir = knowledge_base_dir
        self.vs_dir = vector_store_dir
        self.store_file_path = os.path.join(self.vs_dir, "store.json")

    def _load_documents(self) -> List[Dict[str, str]]:
        """Safely reads all markdown files from the target knowledge base directory."""
        if not os.path.exists(self.kb_dir) or not os.path.isdir(self.kb_dir):
            logging.error(f"[!] Invalid or missing knowledge base directory: {self.kb_dir}")
            return []

        documents = []
        try:
            for filename in sorted(os.listdir(self.kb_dir)):
                # Strict file extension validation
                if not filename.endswith(".md"):
                    continue

                # Defensive path joining to prevent traversal
                filepath = os.path.abspath(os.path.join(self.kb_dir, filename))
                if not filepath.startswith(os.path.abspath(self.kb_dir)):
                    logging.warning(f"Path traversal attempt blocked for file: {filename}")
                    continue

                # File bounds checking to prevent resource exhaustion attacks
                if os.path.getsize(filepath) > MAX_FILE_SIZE_BYTES:
                    logging.warning(f"File {filename} exceeds maximum size limit. Skipping.")
                    continue

                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Format the filename into a clean, human-readable topic
                topic = filename.replace(".md", "").replace("_", " ").title()
                documents.append({
                    "raw_text": content,
                    "filepath": filepath,
                    "topic": topic
                })
        except Exception as e:
            logging.error(f"Error loading documents from knowledge base: {e}")
            
        return documents

    def _parse_markdown_into_chunks(self, document: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Parses a full markdown document into smaller semantic chunks.
        It groups multi-line bullet points or Q&A blocks together for continuity.
        """
        lines = document["raw_text"].strip().split("\n")
        
        doc_title = ""
        doc_desc = ""
        current_block = []
        parsed_blocks = []

        for line in lines:
            stripped_line = line.strip()
            
            # Empty lines signify the end of a block
            if not stripped_line:
                if current_block:
                    parsed_blocks.append(" ".join(current_block))
                    current_block = []
                continue
                
            # Extract header and description specifically
            if stripped_line.startswith("# ") and not doc_title:
                doc_title = stripped_line.lstrip("# ").strip()
            elif stripped_line.startswith("**Description**"):
                doc_desc = stripped_line.replace("**Description**:", "").replace("**Description**", "").strip()
            else:
                # Detect the start of a new numbered point, bullet, or question
                is_new_point = bool(re.match(r"^(\d+[\.\)]|[-*]|Q:)\s", stripped_line))
                
                if is_new_point and current_block:
                    parsed_blocks.append(" ".join(current_block))
                    current_block = [stripped_line]
                else:
                    current_block.append(stripped_line)
        
        # Flush the final block
        if current_block:
            parsed_blocks.append(" ".join(current_block))

        # Build context prefix to artificially boost relevance
        context_prefix = f"Topic: {doc_title}."
        if doc_desc:
            context_prefix += f" {doc_desc}"

        chunks = []
        for block in parsed_blocks:
            chunks.append(self._format_chunk(block, context_prefix, document))

        # Fallback if no specific blocks were detected
        if not chunks:
            chunks.append({
                "text": f"{context_prefix}\n{document['raw_text'].strip()}",
                "source": document['filepath'],
                "topic": document['topic'],
                "term": ""
            })

        return chunks

    def _format_chunk(self, block: str, context_prefix: str, document: Dict[str, str]) -> Dict[str, str]:
        """Cleans formatting artifacts and extracts the primary term for a given block."""
        # Strip numbering and markdown bolding
        clean_block = re.sub(r"^(\d+[\.\)]|[-*]|Q:)\s*", "", block, count=1)
        clean_block = clean_block.replace("**", "")

        chunk_text = f"{context_prefix}\n{clean_block}"

        # Attempt to isolate the specific term being defined (often before a colon)
        term_name = ""
        if block.startswith("Q:"):
             term_match = re.search(r"Q:\s*(.*?)\s*\?", block, re.IGNORECASE)
             if term_match:
                 term_name = term_match.group(1).strip()
             elif ":" in clean_block:
                 term_name = clean_block.split(":")[0].strip()
        elif ":" in clean_block:
             term_name = clean_block.split(":")[0].strip()
            
        # Reject term names that are full sentences
        if len(term_name.split()) > 8:
            term_name = "" 

        return {
            "text": chunk_text,
            "source": document["filepath"],
            "topic": document["topic"],
            "term": term_name
        }

    def _compute_bm25_idf(self, documents_tokens: List[List[str]]) -> Dict[str, float]:
        """Calculates the BM25 specific Inverse Document Frequency for all terms."""
        n_docs = len(documents_tokens)
        doc_frequencies = Counter()
        
        # Count in how many documents each unique term appears
        for tokens in documents_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_frequencies[token] += 1
        
        # Apply smoothing to BM25 formula to prevent negative values
        idf_scores = {}
        for word, count in doc_frequencies.items():
            numerator = n_docs - count + 0.5
            denominator = count + 0.5
            smoothed_score = math.log((numerator / denominator) + 1.0)
            idf_scores[word] = smoothed_score if smoothed_score > 0 else 0.01
            
        return idf_scores

    def build_store(self):
        """Orchestrates the entire indexing procedure and writes the JSON store."""
        os.makedirs(self.vs_dir, exist_ok=True)

        print("[*] Rebuilding vector store from knowledge_base/...")
        documents = self._load_documents()
        if not documents:
            print("[!] No valid markdown files found. Aborting build.")
            return

        all_chunks = []
        for doc in documents:
            all_chunks.extend(self._parse_markdown_into_chunks(doc))
        
        # Tokenize and compute frequencies
        all_tokens = [tokenize_text(chunk["text"]) for chunk in all_chunks]
        idf_scores = self._compute_bm25_idf(all_tokens)

        doc_term_counts = []
        doc_lengths = []
        total_length = 0
        
        for tokens in all_tokens:
            counts = compute_term_frequencies(tokens)
            length = len(tokens)
            doc_term_counts.append(counts)
            doc_lengths.append(length)
            total_length += length
            
        average_doc_length = total_length / max(1, len(all_chunks))

        store_data = {
            "chunks": all_chunks,
            "idf": idf_scores,
            "doc_term_counts": doc_term_counts,
            "doc_lengths": doc_lengths,
            "avgdl": average_doc_length
        }

        with open(self.store_file_path, "w", encoding="utf-8") as f:
            json.dump(store_data, f, ensure_ascii=False, indent=2)

        print(f"[*] Build complete: {len(all_chunks)} chunks indexed successfully.")


def build_vector_store():
    """Entry point for standalone execution."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_dir = os.path.join(base_dir, "knowledge_base")
    vs_dir = os.path.join(base_dir, "vector_store")
    
    indexer = KnowledgeBaseIndexer(kb_dir, vs_dir)
    indexer.build_store()

if __name__ == "__main__":
    build_vector_store()
