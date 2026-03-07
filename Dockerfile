FROM python:3.10-slim

WORKDIR /app

COPY . .

# Generate the vector store embeddings
RUN python scripts/create_embeddings.py

# Run the CLI chatbot
CMD ["python", "app.py"]
