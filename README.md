# Cerevyn Document Intelligence – AI PDF/Q&A Agent

This is a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about their content. The system extracts text from PDFs, splits it into chunks, embeds them using Hugging Face models, stores them in a FAISS vector database, and uses a local LLM to generate answers with page references.

## Features
- Upload PDF files and extract text with page numbers
- Split text into manageable chunks for better retrieval
- Embed chunks using sentence transformers
- Store embeddings in FAISS for fast similarity search
- Generate answers using GPT-2 model with context from retrieved chunks
- Display answers with references to source pages
- Support for multiple documents in session
- Clean and simple Streamlit web interface

## Setup Instructions
1. Clone this repository to your local machine.
2. Create a virtual environment: `python -m venv env`
3. Activate the environment: `env\Scripts\activate` (Windows) or `source env/bin/activate` (Linux/Mac)
4. Install the required packages: `pip install -r requirements.txt`
5. Run the application: `streamlit run app.py`

## Usage Guide
1. Open the app in your browser after running the command above.
2. Upload a PDF file using the file uploader.
3. Click the "Process PDF" button to extract text and build the vector index.
4. Enter a question in the text input field.
5. Click "Ask" to get an answer based on the document content.
6. View the answer and the source pages where the information was found.

## Architecture Overview
The system follows a standard RAG pipeline:
- **Text Extraction**: Uses pdfplumber to extract text from each page of the PDF.
- **Text Splitting**: Splits the extracted text into overlapping chunks using RecursiveCharacterTextSplitter.
- **Embedding**: Converts text chunks into vector embeddings using HuggingFaceEmbeddings.
- **Vector Storage**: Stores embeddings in FAISS for efficient similarity search.
- **Retrieval**: Retrieves top-k similar chunks based on the question embedding.
- **Generation**: Uses GPT-2 to generate answers from the retrieved context.
- **Interface**: Streamlit provides a web UI for interaction.

See `architecture_diagram.txt` for a visual representation.

## Deployment
To deploy this app on Streamlit Cloud:
1. Push this code to a GitHub repository.
2. Go to share.streamlit.io and connect your GitHub account.
3. Select the repository and set the main file path to `app.py`.
4. Deploy the app.

No API keys are required since it uses local models.
