# Hellobooks AI — RAG-Based Accounting Assistant

An AI-powered bookkeeping assistant that answers accounting-related questions using a **Retrieval-Augmented Generation (RAG)** pipeline built entirely in Python.

## Project Overview

Hellobooks AI is a prototype intelligent accounting assistant. It retrieves relevant information from a curated knowledge base and generates accurate, context-aware answers about accounting concepts.

### RAG Architecture

```
User Question → Retrieve relevant documents → Send context to LLM → Generate answer
```

1. **Document Loading** — Markdown files from `knowledge_base/` are loaded and split into chunks.
2. **Embedding Generation** — Each chunk is converted into a TF-IDF vector (term frequency–inverse document frequency).
3. **Vector Store** — All vectors are stored locally in a JSON file (`vector_store/store.json`).
4. **Retrieval** — The user's question is vectorized and compared against stored chunks using cosine similarity.
5. **Answer Generation** — The most relevant chunks are combined into a coherent answer.

## Dataset Explanation

The `knowledge_base/` folder contains **5 markdown documents** covering core accounting topics:

| File | Topic |
|------|-------|
| `bookkeeping.md` | Recording transactions, ledger, trial balance, accounts payable/receivable |
| `invoices.md` | Invoice structure, billing, tax, payment terms |
| `profit_loss.md` | Revenue, expenses, gross/net profit, operating costs |
| `balance_sheet.md` | Assets, liabilities, equity, financial position |
| `cash_flow.md` | Cash inflow/outflow, operating/investing/financing activities |

## Installation Instructions

```bash
git clone <your-repo-url>
cd hellobooks-ai
```

No external packages are required — the project uses **pure Python** (3.10+).

## How to Generate Embeddings

Before using the chatbot, you must build the vector store:

```bash
python scripts/create_embeddings.py
```

This reads all knowledge base documents, splits them into chunks, computes TF-IDF embeddings, and saves them to `vector_store/store.json`.

## How to Run the Chatbot

```bash
python app.py
```

Type your accounting question and press Enter. Type `exit` or `quit` to close.

## How to Run Using Docker

Build the image:

```bash
docker build -t hellobooks-ai .
```

Run the container:

```bash
docker run -it hellobooks-ai
```

## Example Questions

- "What is a balance sheet?"
- "Explain double entry accounting"
- "What are the components of an invoice?"
- "How is net profit calculated?"
- "What is cash flow from operating activities?"

## Example Commands

```bash
python scripts/create_embeddings.py
python app.py
```

## Project Structure

```
hellobooks-ai/
├── knowledge_base/
│   ├── bookkeeping.md
│   ├── invoices.md
│   ├── profit_loss.md
│   ├── balance_sheet.md
│   └── cash_flow.md
├── scripts/
│   ├── create_embeddings.py
│   └── rag_pipeline.py
├── vector_store/
│   └── store.json
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```
