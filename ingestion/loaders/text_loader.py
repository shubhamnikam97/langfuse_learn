from typing import List
from ingestion.loaders.base_loader import BaseLoader

class TextLoader(BaseLoader):

    def load(self, file_path:str) -> List[str]:
        with open(file_path, "r", encoding='utf-8') as f:
            return [f.read()]