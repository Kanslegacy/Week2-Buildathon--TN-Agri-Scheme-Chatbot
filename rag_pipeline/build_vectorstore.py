import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()  # reads OPENAI_API_KEY from your .env file

INPUT_FILE = "rag_pipeline/cleaned_documents.json"
VECTORSTORE_DIR = "rag_pipeline/faiss_index"

def load_documents():
    """Convert our saved JSON into LangChain Document objects."""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    documents = []
    for item in raw_docs:
        doc = Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        documents.append(doc)
    return documents

def main():
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    # 'text-embedding-3-small' is a low-cost, lightweight embedding model
    # — perfect for a project with only 54 short documents.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Generating embeddings and building FAISS index... (this calls the OpenAI API)")
    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"✅ FAISS vector store saved to: {VECTORSTORE_DIR}")

if __name__ == "__main__":
    main()
