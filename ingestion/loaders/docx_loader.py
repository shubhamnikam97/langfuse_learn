from typing import List
from docx import Document
from ingestion.loaders.base_loader import BaseLoader

class DocxLoader(BaseLoader):
    def load(self, file_path:str) -> List[str]:
        doc = Document(file_path)
        texts = [para.text for para in doc.paragraphs if para.text.strip()]
        return ["\n".join(texts)]