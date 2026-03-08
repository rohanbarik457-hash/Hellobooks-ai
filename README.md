# Hellobooks AI — RAG-Based Accounting Assistant

An AI-powered bookkeeping assistant that answers accounting-related questions using a **Retrieval-Augmented Generation (RAG)** pipeline built entirely in Python.

## Project Overview

Hellobooks AI is a prototype intelligent accounting assistant. It retrieves relevant information from a curated knowledge base of **500 detailed accounting points** and generates accurate, context-aware answers.

### RAG Architecture

```
User Question → Security Validation → Retrieve BM25 Context → Send to Interface → Generate Answer
```

1. **Document Loading** — Markdown files from `knowledge_base/` are securely loaded (with path traversal and size limits) and split into fine-grained chunks.
2. **Embedding Generation** — Each chunk is converted into an intelligent **BM25 statistical index** (pure Python) with term-boosting capabilities.
3. **True Live-Reload** — The system automatically detects knowledge base updates per query and seamlessly rebuilds the vector store instantly without requiring a restart.
4. **Vector Store** — All vectors are stored locally in a JSON file (`vector_store/store.json`).
5. **Retrieval** — Uses the advanced BM25 ranking algorithm to prioritize exact definition matches over partial occurrences.
6. **Answer Generation** — Extractive generation synthesizes relevant chunks into a clean, numbered answer.

## Cybersecurity Hardening
This pipeline has been refactored explicitly for production security:
- **Resource Exhaustion (DoS) Protection:** Hard limits on query lengths and parsed file sizes.
- **Directory Traversal Blocking:** Strict `.realpath()` mathematical proofs prevent arbitrary file reads.
- **Opaque Handling:** Errors log privately to an internal system file without leaking tracebacks to the end-user.
## Dataset Explanation

The `knowledge_base/` folder contains **500 items** across 5 topics, each structured with a **Definition** and a **Real-World Example**:

| Topic | Entries | Description |
|-------|---------|-------------|
| **Bookkeeping** | 100 | Recording transactions, double-entry, ledgers, journals, etc. |
| **Invoices** | 100 | Billing terms, itemized billing, taxes, payment cycles, etc. |
| **Profit & Loss** | 100 | Revenue, COGS, gross/net profit, operating expenses, etc. |
| **Balance Sheet** | 100 | Assets, liabilities, equity, financial ratios, liquidity, etc. |
| **Cash Flow** | 100 | Operating/investing/financing activities, cash inflow/outflow, etc. |

## Installation Instructions

1. **Clone the project:**
   ```bash
   git clone <your-repo-url>
   cd hellobooks-ai
   ```

2. **Python Environment:**
   No external packages are required — the project uses **pure Python** (3.10+).

## How to Run the Chatbot

The system features **Automatic Embedding Generation**. It will build the vector store on first run or whenever you update the documents.

```bash
python app.py
```

- Type your accounting question and press Enter.
- Type `exit` or `quit` to close.

## How to Run Using Docker

1. **Build the image:**
   ```bash
   docker build -t hellobooks-ai .
   ```

2. **Run the container:**
   ```bash
   docker run -it hellobooks-ai
   ```

## Example Questions

- "What is job order costing?"
- "Tell me about the lockbox system."
- "What is a pro-forma invoice?"
- "What is inventory shrinkage?"
- "Explain shareholder value with an example."

## Project Structure

```
hellobooks-ai/
├── knowledge_base/        # Curated generic markdown datasets
├── scripts/               # Object-Oriented Backend
│   ├── create_embeddings.py # Vector generation & parsing logic
│   ├── rag_pipeline.py      # Core BM25 query & live-sync orchestrator
│   └── text_processing.py   # Tokenization utility class
├── vector_store/
│   └── store.json         # Local vector database
├── system_errors.log      # Secure internal error logging
├── app.py                 # Hardened CLI Interface
├── requirements.txt       # Minimal dependencies
├── Dockerfile             # Containerization
└── README.md              # Project documentation
```
