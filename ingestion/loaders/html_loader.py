from typing import List
from bs4 import BeautifulSoup

from ingestion.loaders.base_loader import BaseLoader


class HTMLLoader(BaseLoader):
    def load(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        return [text]