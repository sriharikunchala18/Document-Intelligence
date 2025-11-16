import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from transformers import pipeline
from utils import extract_text_from_pdf, split_text_with_metadata

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
pipe = pipeline("text-generation", model="gpt2", max_new_tokens=100, temperature=0.1, device=-1)
llm = HuggingFacePipeline(pipeline=pipe)

st.title("Cerevyn Document Intelligence – AI PDF/Q&A Agent")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents" not in st.session_state:
    st.session_state.documents = []

# PDF upload section
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

# Q&A section
if st.session_state.vectorstore:
    st.subheader("Ask a question about the uploaded documents")
    question = st.text_input("Enter your question:")
    if question and st.button("Ask"):
        with st.spinner("Generating answer..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
            context = "\n".join([doc.page_content[:300] for doc in docs])
            prompt = f"Answer the question based only on the provided context. If the answer is not in the context, say 'I don't know'.\nContext: {context}\nQuestion: {question}\nAnswer:"
            response = llm.invoke(prompt)
            if isinstance(response, list):
                full_text = response[0].get('generated_text', str(response[0]))
            else:
                full_text = str(response)
            if full_text.startswith(prompt):
                answer = full_text[len(prompt):].strip()
            else:
                answer = full_text.strip()
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
