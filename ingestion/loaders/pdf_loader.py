from typing import List, Dict, Any
from ingestion.loaders.base_loader import BaseLoader
from pypdf import PdfReader
import fitz # PyMuPDF # for image extraction
import pdfplumber # for table extraction
import os 
import uuid

class PDFLoader(BaseLoader):
    def load(self, file_path:str) -> List[Dict[str, Any]]:

        """
        Returns multimodal content:
        [
            {"type": "text", "content": "...", "page": 1},
            {"type": "image", "content": bytes, "page": 1}
        ]
        """
        results = []

        # -----------------------
        # Text
        # -----------------------
        reader = PdfReader(file_path)
        
        # for i, page in enumerate(reader.pages, start=1):
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                results.append(
                    {'type': "text",
                     'content': text,
                     'page': i}
                )

        # -----------------------
        # Tables
        # -----------------------
        results.extend(self.extract_tables(file_path))

        # -----------------------
        # Images
        # -----------------------
        if fitz:
            results.extend(self.extract_images(file_path))

        return results
    
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        tables_data = []

        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                tables = page.extract_tables()

            for table in tables:
                if not table:
                    continue

                table_text = "\n".join(
                    [" | ".join([str(cell) for cell in row]) for row in table]
                    )

                tables_data.append({
                    "type": "table",
                    "content": table_text,
                    "page": page_index
                })

        return tables_data
    
    def extract_images(self, file_path: str) -> List[Dict[str, Any]]:

        images_data = []

        doc = fitz.open(file_path)

        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)

            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                images_data.append({
                    "type": "image",
                    "content": image_bytes,
                    "page": page_index
                })

        return images_data
        
    
