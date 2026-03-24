# Mini RAG: Construction Marketplace Assistant

This repository contains a Retrieval-Augmented Generation (RAG) pipeline built for a construction marketplace.

## Architecture & Choices
* **Embedding Model:** `all-MiniLM-L6-v2` (via `sentence-transformers`). Chosen because it is a lightweight, highly efficient open-source model perfect for local document embedding without external API latency.
* **Vector Store:** FAISS. Chosen for its speed and efficiency in local similarity search using L2 distance.
* **LLM:** `meta-llama/llama-3-8b-instruct:free` (via OpenRouter). Chosen to fulfill the free LLM requirement while providing strong instruction-following capabilities to prevent hallucinations.

## Implementation Details
1. **Chunking:** Documents are parsed from the `documents/` folder using `PyPDF2`. The text is split into fixed-size strings (500 characters) with a 50-character overlap to preserve semantic meaning across boundaries.
2. **Retrieval:** User queries are embedded using the same `MiniLM` model. FAISS performs a semantic similarity search to retrieve the top-3 most relevant chunks.
3. **Grounding:** The LLM is explicitly prompted via system instructions to answer *only* using the provided context, enforcing strict grounding and preventing unsupported claims.

## How to Run Locally
1. Clone the repository and navigate to the folder.
2. Place the assessment PDFs inside the `documents/` directory.
3. Create a virtual environment and activate it.
4. Run `pip install -r requirements.txt`
5. Replace `YOUR_OPENROUTER_API_KEY` in `rag_pipeline.py` with your actual key.
6. Run the application: `streamlit run app.py`