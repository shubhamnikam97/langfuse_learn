from abc import ABC, abstractmethod
from typing import List

class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    """

    @abstractmethod
    def load(self, file_path: str) -> List[str]:
        """
        Load file and return list of documents(string)"""
        pass