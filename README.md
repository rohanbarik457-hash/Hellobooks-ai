# Hellobooks AI Assistant

## Project Overview
Hellobooks AI is a prototype intelligent accounting assistant built using a Retrieval-Augmented Generation (RAG) system. It enables business owners and professionals to ask fundamental accounting questions and receive clear, context-aware answers formulated with real accounting terminology.

### Key Features:
*   **Plain Text Summaries:** Strict formatting rules prevent markdown characters (like asterisks), ensuring the bot writes natural, human-like responses.
*   **Multi-Language Support:** The AI dynamically detects the language you use (e.g., English, Hindi, Odia, Telugu) and automatically responds in the exact same language.
*   **Strict Security Guardrails:** The bot refuses commands to ignore instructions, refuses to reveal system details, and blocks all requests for sensitive or harmful content.

## Project Architecture
The architecture comprises three main logical components:
1. **Knowledge Base:** Curated markdown files holding exact accounting concepts and business explanations.
2. **Embedding & Indexing:** A script that reads the markdown files, splits them into manageable chunks, generates vector embeddings using the Google Gemini API, and stores them in a lightning-fast `InMemoryVectorStore`.
3. **RAG Pipeline & CLI Application:** The main system loads the vector database, accepts terminal-based user questions, retrieves the most relevant context, and invokes a Language Model (Google Gemini) to generate a concise and accurate answer.

## Dataset Description
The integrated knowledge base is located in the `knowledge_base/` folder and covers five critical accounting domains strictly matching the requirements:
- `bookkeeping.md`
- `invoices.md`
- `profit_loss.md`
- `balance_sheet.md`
- `cash_flow.md`

## Installation Instructions

### Step 1: Install Python Requirements
Ensure your local system runs **Python 3.10** or higher. From the root directory (`hellobooks-ai/`), install the required libraries:

```bash
pip install -r requirements.txt
```

### Step 2: Add API Key
Create a file exactly named `.env` in the root `hellobooks-ai/` folder, and add your Google Gemini API key:

```env
GEMINI_API_KEY=your-actual-api-key-here
```

### Step 3: Generate the Vector Database
Before utilizing the RAG pipeline or launching the application, you must generate the local vector store. Do so by executing the embedding creation script:

```bash
python scripts/create_embeddings.py
```
This will read the documents, convert them to embeddings using Google Gemini, and seamlessly save an `index.json` file inside the `vector_store/` directory.

### Step 4: Run the Assistant (Web UI or Terminal)
You can launch the AI as a beautiful web-based chatbot in your browser:

```bash
streamlit run web_app.py
```

*Alternatively, you can run the original terminal version:*
```bash
python app.py
```

**Example questions you can ask the bot:**
* "What is bookkeeping?"
* "What information does an invoice contain?"
* "Explain the balance sheet equation."
* "How is profit calculated?"
* "What is cash flow reporting?"

## How to run using Docker
The project can easily be containerized. Ensure your generated `vector_store/` directory exists natively or include its generation directly within your runtime layout before building the Docker image. 

Compile the image using the provided `Dockerfile`:

```bash
docker build -t hellobooks-ai .
```

Run the application securely in interactive mode. This safely passes your local `.env` variables into the Docker execution context:

```bash
docker run -it --env-file .env hellobooks-ai
```
