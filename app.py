import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.documents import Document
from utils import extract_text_from_pdf, split_text_with_metadata

load_dotenv()

openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set OPENAI_API_KEY in Streamlit Cloud secrets or .env file")
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
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 most relevant chunks
            docs = retriever.invoke(question)
            # Limit context to avoid token limit
            context = "\n".join([doc.page_content[:300] for doc in docs])  # 300 chars per chunk
            prompt = f"Answer the question based only on the provided context. If the answer is not in the context, say 'I don't know'.\nContext: {context}\nQuestion: {question}\nAnswer:"
            response = llm.invoke(prompt)
            # Extract text from response
            if isinstance(response, list):
                full_text = response[0].get('generated_text', str(response[0]))
            else:
                full_text = str(response)
            # Remove the prompt from the response
            if full_text.startswith(prompt):
                answer = full_text[len(prompt):].strip()
            else:
                answer = full_text.strip()
            # Clean up answer: take first line, remove extra text
            answer = answer.split('\n')[0].strip()
            if not answer or len(answer) < 5 or answer.lower().startswith("answer:") or "context:" in answer.lower():
                answer = "I don't know based on the document."
            sources = docs
            st.write("**Answer:**", answer)
            st.write("**Sources:**")
            for source in sources:
                st.write(f"- Page {source.metadata['page']}: {source.page_content[:200]}...")
else:
    st.info("Please upload and process a PDF to start asking questions.")
