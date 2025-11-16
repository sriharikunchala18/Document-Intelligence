import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from langchain_core.documents import Document
from transformers import pipeline
from utils import extract_text_from_pdf, split_text_with_metadata

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use a better Hugging Face model for LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.1, "max_new_tokens": 300, "do_sample": True, "pad_token_id": 50256}
)

st.title("Cerevyn Document Intelligence – AI PDF/Q&A Agent")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents" not in st.session_state:
    st.session_state.documents = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.button("Process PDF"):
    with st.spinner("Processing PDF..."):
        pages = extract_text_from_pdf(uploaded_file)
        chunks = split_text_with_metadata(pages)
        docs = [Document(page_content=chunk["text"], metadata={"page": chunk["page"]}) for chunk in chunks]
        st.session_state.documents.extend(docs)
        if st.session_state.vectorstore:
            st.session_state.vectorstore.add_documents(docs)
        else:
            st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    st.success("PDF processed successfully!")

if st.session_state.vectorstore:
    st.subheader("Ask a question about the uploaded documents")
    question = st.text_input("Enter your question:")
    if question and st.button("Ask"):
        with st.spinner("Generating answer..."):
            retriever = st.session_state.vectorstore.as_retriever()
            docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            answer = llm(prompt)
            sources = docs
            st.write("**Answer:**", answer)
            st.write("**Sources:**")
            for source in sources:
                st.write(f"- Page {source.metadata['page']}: {source.page_content[:200]}...")
else:
    st.info("Please upload and process a PDF to start asking questions.")
