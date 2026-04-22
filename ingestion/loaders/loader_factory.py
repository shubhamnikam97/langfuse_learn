import os

from ingestion.loaders.text_loader import TextLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.docx_loader import DocxLoader

def get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return TextLoader()
    elif ext == ".pdf":
        return PDFLoader()
    elif ext == ".docx":
        return DocxLoader()
    else:
        raise ValueError(f"Unsupported file type: {ext}")