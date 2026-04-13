"""Generation component for RAG."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

from .retriever import Retriever
from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Generator:
    """Handles text generation for RAG responses."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            openai_api_key=settings.openai_api_key
        )

        # Define the RAG prompt template
        self.qa_prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

        # Create the RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever.vectorstore_manager.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )

    def generate_answer(self, query: str, use_compression: bool = False) -> Dict[str, Any]:
        """Generate an answer for the given query using RAG."""
        try:
            # Retrieve relevant documents
            docs = self.retriever.retrieve(query, use_compression=use_compression)

            # Generate answer using the QA chain
            result = self.qa_chain({"query": query})

            response = {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in result["source_documents"]
                ],
                "query": query
            }

            logger.info(f"Generated answer for query: {query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while processing your query.",
                "error": str(e),
                "query": query
            }

    def generate_streaming_answer(self, query: str) -> Dict[str, Any]:
        """Generate a streaming answer (placeholder for future implementation)."""
        # This would implement streaming responses
        return self.generate_answer(query)