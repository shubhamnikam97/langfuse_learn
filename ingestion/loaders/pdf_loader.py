from typing import List
from ingestion.loaders.base_loader import BaseLoader
from pypdf import PdfReader

class PDFLoader(BaseLoader):
    def load(self, file_path:str) -> List[str]:
        reader = PdfReader(file_path)
        
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

        return texts