import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import requests
st.secrets["OPENROUTER_API_KEY"]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"

model = SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdfs(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(directory, filename))
            for page in reader.pages:
                text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def build_vector_store(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def retrieve_context(query, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def generate_answer(query, context):
    prompt = f"""
    Answer the following question STRICTLY using the provided context. 
    If the answer is not contained in the context, say "I don't know based on the documents."
    Do not use outside knowledge.
    
    Context:
    {context}
    
    Question: {query}
    """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "meta-llama/llama-3-8b-instruct:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']
