import streamlit as st
from rag_pipeline import extract_text_from_pdfs, chunk_text, build_vector_store, retrieve_context, generate_answer

st.set_page_config(page_title="Construction Assistant", layout="wide")
st.title("Construction Marketplace AI Assistant")

@st.cache_resource
def initialize_rag():
    raw_text = extract_text_from_pdfs("documents")
    doc_chunks = chunk_text(raw_text)
    faiss_index, _ = build_vector_store(doc_chunks)
    return doc_chunks, faiss_index

chunks, index = initialize_rag()

query = st.text_input("Ask a question about construction policies or specs:")

if st.button("Submit"):
    if query:
        with st.spinner("Retrieving and generating..."):
            retrieved_chunks = retrieve_context(query, index, chunks)
            context_string = "\n\n".join(retrieved_chunks)
            
            answer = generate_answer(query, context_string)
            
            st.subheader("Final Answer")
            st.write(answer)
            
            st.subheader("Retrieved Context (Transparency)")
            for i, chunk in enumerate(retrieved_chunks):
                st.info(f"Chunk {i+1}:\n{chunk}")