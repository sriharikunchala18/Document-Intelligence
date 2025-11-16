# Cerevyn Document Intelligence – AI PDF/Q&A Agent

A RAG-based system for uploading PDFs and asking questions about their content.

## Features
- PDF upload and text extraction
- Vector search using FAISS
- Q&A with page references
- Support for multiple documents
- Clean Streamlit UI

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. For local development, set up OpenAI API key in `.env` file: `OPENAI_API_KEY=your_key_here`
4. For Streamlit Cloud deployment, add `OPENAI_API_KEY` to secrets.
5. Run the app: `streamlit run app.py`

## Usage
1. Upload a PDF file.
2. Click "Process PDF" to extract and index the text.
3. Ask questions in the chat interface.
4. Get answers with source page references.

## Architecture
See `architecture_diagram.txt` for a simple diagram.

## Deploy
Deploy to Streamlit Cloud by connecting your GitHub repo.
