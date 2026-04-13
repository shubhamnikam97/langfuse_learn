# RAG Application

A production-grade Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, and integrated with Langfuse for monitoring.

## Features

- **Document Ingestion**: Support for PDF, TXT, and CSV files
- **Vector Storage**: ChromaDB and FAISS support
- **Retrieval**: Similarity search with optional compression
- **Generation**: OpenAI GPT integration for answer generation
- **API**: RESTful API with FastAPI
- **Monitoring**: Langfuse integration for observability
- **Containerization**: Docker support

## Project Structure

```
├── src/rag_app/
│   ├── api/           # FastAPI application
│   ├── config/        # Configuration settings
│   ├── core/          # Core RAG components (retrieval, generation, vectorstore)
│   ├── data/          # Data ingestion and preprocessing
│   └── utils/         # Utility functions
├── tests/             # Unit and integration tests
├── scripts/           # Utility scripts
├── config/            # Configuration files
├── docs/              # Documentation
├── docker/            # Docker-related files
├── data/              # Data storage
├── models/            # Model artifacts
├── main.py            # Application entry point
├── pyproject.toml     # Project dependencies
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Docker Compose configuration
└── .env.example       # Environment variables template
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Copy environment file and configure:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit the `.env` file with your configuration:

- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGFUSE_PUBLIC_KEY`: Langfuse public key (optional)
- `LANGFUSE_SECRET_KEY`: Langfuse secret key (optional)

## Usage

### Running Locally

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Using Docker

```bash
docker-compose up --build
```

### API Endpoints

- `GET /`: Health check
- `POST /query`: Query the RAG system
- `POST /ingest`: Ingest documents
- `GET /documents`: List documents

### Example API Usage

```python
import requests

# Query the system
response = requests.post("http://localhost:8000/query",
    json={"query": "What is machine learning?"})
print(response.json())

# Ingest documents
response = requests.post("http://localhost:8000/ingest",
    json={"data_source": "./data/documents"})
print(response.json())
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
```

## Deployment

The application can be deployed using Docker or cloud platforms like AWS, GCP, or Azure.

## Monitoring

When Langfuse credentials are provided, the application will automatically log traces and metrics to Langfuse for monitoring and debugging.

## License

MIT License