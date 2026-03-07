import os
import time
import requests
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning)

class GeminiRestEmbeddings:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            print(f"[*] Embedding chunk {i+1}/{len(texts)}...")
            embeddings.append(self.embed_query(text))
            time.sleep(0.5)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        payload = {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]}}
        headers = {"Content-Type": "application/json"}
        
        for _ in range(3):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=10)
                data = response.json()
                
                if "embedding" in data:
                    return data["embedding"]["values"]
                elif response.status_code == 429:
                    print("Rate limited. Waiting...")
                    time.sleep(2)
                else:
                    raise Exception(f"Failed to generate embedding: {data}")
            except requests.exceptions.Timeout:
                print("Timeout. Retrying...")
                time.sleep(1)
                
        raise Exception("Failed to generate embedding after 3 retries")

def build_vector_store():
    load_dotenv()
    
    api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment.")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_base_path = os.path.join(base_dir, "knowledge_base")
    vector_store_path = os.path.join(base_dir, "vector_store")
    
    os.makedirs(vector_store_path, exist_ok=True)
    store_file = os.path.join(vector_store_path, "index.json")

    print("[*] Loading documents...")
    loader = DirectoryLoader(knowledge_base_path, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[*] Created {len(chunks)} chunks.")

    print("[*] Generating embeddings...")
    embedder = GeminiRestEmbeddings(api_key=api_key)
    
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks, 
        embedding=embedder
    )
    
    vector_store.dump(store_file)
    print(f"[*] Vector store saved to {store_file}")

if __name__ == "__main__":
    build_vector_store()
