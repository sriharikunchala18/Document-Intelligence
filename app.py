import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains.retrieval_qa import RetrievalQA
from langchain_core.documents import Document
from utils import extract_text_from_pdf, split_text_with_metadata

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set OPENAI_API_KEY in .env file")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

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
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(),
                return_source_documents=True
            )
            result = qa_chain({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            st.write("**Answer:**", answer)
            st.write("**Sources:**")
            for source in sources:
                st.write(f"- Page {source.metadata['page']}: {source.page_content[:200]}...")
else:
    st.info("Please upload and process a PDF to start asking questions.")
