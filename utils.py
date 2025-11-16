import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file):
    """
    Extract text from PDF and return list of dicts with page number and text.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pages = []
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append({"page": page_num + 1, "text": text})
    return pages

def split_text_with_metadata(pages, chunk_size=1000, chunk_overlap=200):
    """
    Split text into chunks while preserving page metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page["text"])
        for chunk in page_chunks:
            chunks.append({"text": chunk, "page": page["page"]})
    return chunks
