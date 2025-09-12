import os
from pathlib import Path
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Configure paths
PDF_PATH = "data/softwayLabpdf2.pdf"
INDEX_PATH = "faiss_index/index.faiss"
DOCUMENTS_PATH = "faiss_index/documents.npy"

# Ensure output directory exists
Path("faiss_index").mkdir(parents=True, exist_ok=True)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
text = ""
try:
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
except Exception as e:
    print(f"Error extracting PDF: {e}")
    exit(1)

if not text.strip():
    print("No text content found in PDF")
    exit(1)

# Chunk text
def chunk_text(text, chunk_size=500, overlap=100):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for i, sentence in enumerate(sentences):
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentences[max(0, i - overlap // 50)] + ". " if i > 0 else sentence + ". "
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if len(chunk) > 50]

chunks = chunk_text(text)
if not chunks:
    print("No valid chunks created from PDF")
    exit(1)

# Create document list with dictionary format
documents = [{"content": chunk} for chunk in chunks]

# Generate embeddings
embeddings = model.encode([doc["content"] for doc in documents])
dim = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save index and documents
faiss.write_index(index, INDEX_PATH)
np.save(DOCUMENTS_PATH, np.array(documents, dtype=object))

print(f"FAISS index created with {len(documents)} documents and saved to {INDEX_PATH}")
print(f"Documents saved to {DOCUMENTS_PATH}")