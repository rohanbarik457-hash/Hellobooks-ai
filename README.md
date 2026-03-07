# Hellobooks AI — RAG-Based Accounting Assistant

An AI-powered bookkeeping assistant that answers accounting-related questions using a **Retrieval-Augmented Generation (RAG)** pipeline built entirely in Python.

## Project Overview

Hellobooks AI is a prototype intelligent accounting assistant. It retrieves relevant information from a curated knowledge base of **500 detailed accounting points** and generates accurate, context-aware answers.

### RAG Architecture

```
User Question → Retrieve relevant documents → Send context to LLM → Generate answer
```

1. **Document Loading** — Markdown files from `knowledge_base/` are loaded and split into 500 fine-grained chunks.
2. **Embedding Generation** — Each chunk is converted into a TF-IDF vector (pure Python).
3. **Auto-Sync** — The system automatically detects knowledge base updates and rebuilds the vector store.
4. **Vector Store** — All vectors are stored locally in a JSON file (`vector_store/store.json`).
5. **Retrieval** — Uses cosine similarity to find the most relevant definitions and examples.
6. **Answer Generation** — Extractive generation synthesizes relevant chunks into a clear answer.

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
├── knowledge_base/        # 500-item accounting dataset
├── scripts/
│   ├── create_embeddings.py # Vector generation logic
│   └── rag_pipeline.py     # RAG & Auto-sync logic
├── vector_store/
│   └── store.json         # Local vector database
├── app.py                 # CLI Interface
├── requirements.txt       # Minimal dependencies
├── Dockerfile             # Containerization
└── README.md              # Project documentation
```
