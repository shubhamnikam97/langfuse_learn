# RAG Application

A production-grade Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, and integrated with Langfuse for monitoring. Features comprehensive document ingestion, multi-strategy retrieval, intelligent reranking, quality evaluation, and guardrails enforcement—all containerized with Docker for seamless deployment.

## Features

- **Multi-Format Document Ingestion**: PDF, DOCX, HTML, TXT file support with automatic processing
- **Advanced Text Processing**: Recursive chunking, metadata extraction, and text cleaning
- **Multi-Engine Vector Storage**: ChromaDB and FAISS support with flexible index management
- **Intelligent Retrieval**: Hybrid search strategies with query transformation and multi-query support
- **Smart Reranking**: Optional document reranking to improve retrieval quality
- **Dual Interface**: RESTful FastAPI backend + Streamlit UI for interactive querying
- **LLM Generation**: OpenAI GPT-4o integration with prompt templating and response optimization
- **Quality Evaluation**: RAGAS framework integration for faithfulness, relevancy, and context precision metrics
- **Guardrails & Safety**: Input validation and output constraints to ensure safe, reliable responses
- **Feedback Collection**: User feedback mechanism for continuous improvement
- **Analytics & Insights**: Analytics module for tracking usage patterns and performance
- **Production Monitoring**: Langfuse integration for observability, logging, and cost tracking
- **Containerization**: Complete Docker setup for reproducible deployment

## Project Structure

```
rag-project/
│
├── app/                         # Application layer (UI / API)
│   ├── api/                     # FastAPI / Flask routes
│   │   ├── routes.py
│   │   ├── dependencies.py
│   │   └── schemas.py
│   │
│   ├── ui/                      # Streamlit / frontend
│   │   └── app.py
│   │
│   └── main.py                  # Entry point
│
├── core/                        # Core configs & settings
│   ├── config.py                # env configs
│   ├── constants.py
│   ├── langfuse_client.py
│   └── logging.py
│
├── ingestion/                   # Data ingestion pipeline
│   ├── loaders/
│   │   ├── loader_factory.py    # Factory pattern for loader selection
│   │   ├── base_loader.py       # Base loader interface
│   │   ├── pdf_loader.py
│   │   ├── docx_loader.py
│   │   ├── html_loader.py
│   │   └── text_loader.py
│   │
│   ├── processors/
│   │   ├── text_cleaner.py
│   │   ├── chunker.py           # recursive splitter logic
│   │   └── metadata_extractor.py
│   │
│   └── pipeline.py              # ingestion pipeline orchestration
│
├── embedding/                  # Embedding generation & management
│   ├── client.py                # Embedding API client
│   ├── embedding_model.py       # Embedding model wrapper
│   ├── embedding_pipleline.py   # Embedding generation pipeline
│   └── pipeline.py              # Pipeline orchestration
│
├── vectorstore/                 # Vector database abstraction layer
│   ├── base.py                  # Base vector store interface
│   ├── factory.py               # Factory for vector store selection
│   ├── chroma_store.py          # ChromaDB implementation
│   ├── faiss_store.py           # FAISS implementation
│   ├── vector_client.py         # Vector store client wrapper
│   ├── index_manager.py         # Index creation & management
│   └── retriever.py             # Vector-based retrieval
│
├── retrieval/                   # Retrieval strategies & optimization
│   ├── base_retriever.py        # Base retriever interface
│   ├── retriever.py             # Main retriever implementation
│   ├── hybrid_retriever.py      # Hybrid search combining BM25 & semantic search
│   ├── reranker.py              # Document reranking for quality
│   └── query_transform.py       # Query expansion & transformation
│
├── llm/                         # LLM integration & response generation
│   ├── llm_client.py            # OpenAI API wrapper
│   ├── generator.py             # Core generation logic
│   ├── prompt_templates.py      # Prompt templates for various tasks
│   └── response_generator.py    # Response formatting & optimization
│
├── rag/                         # RAG orchestration layer
│   ├── pipeline.py              # main RAG pipeline
│   ├── context_builder.py
│   └── evaluator.py             # eval metrics (faithfulness, etc.)
│
├── evaluation/                  # Quality evaluation framework
│   └── ragas_evaluator.py       # RAGAS metrics (faithfulness, relevancy, precision)
│
├── guardrails/                  # Safety & validation layer
│   └── [guardrails config]      # Input/output constraints & validation
│
├── feedback/                    # User feedback collection
│   └── [feedback handlers]      # Feedback storage and analysis
│
├── analytics/                   # Usage & performance analytics
│   └── [analytics modules]      # Metrics, dashboards, insights
│
├── agents/                      # Agentic workflows (optional)
│   └── [agent implementations]  # Multi-step reasoning & tool use
│
├── storage/                     # File & object storage
│   ├── gcs_client.py            # Google Cloud Storage
│   └── document_store.py
│
├── utils/                       # Utility functions
│   ├── helpers.py
│   ├── tokenizer.py
│   └── validators.py
│
├── tests/                       # Unit & integration tests
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_rag_pipeline.py
│
├── scripts/                     # CLI scripts
│   ├── ingest_data.py
│   ├── reindex.py
│   └── evaluation.py
│
├── configs/                     # YAML configs
│   ├── model_config.yaml
│   ├── pipeline_config.yaml
│   └── retriever_config.yaml
│
├── docker/                      # Containerization
│   └── Dockerfile
│
├── .env
├── requirements.txt
├── pyproject.toml               # better dependency mgmt
├── README.md
└── Makefile                     # automation
```

## Key Workflows

### Document Ingestion Pipeline
1. **Upload** → Documents (PDF, DOCX, HTML, TXT) → `ingestion/loaders/`
2. **Process** → Text cleaning, chunking, metadata extraction → `ingestion/processors/`
3. **Embed** → Generate embeddings for each chunk → `embedding/`
4. **Store** → Index vectors in ChromaDB/FAISS → `vectorstore/`

### Query & Response Pipeline
1. **Receive Query** → FastAPI/Streamlit interface → `app/`
2. **Transform** → Multi-query expansion, normalization → `retrieval/query_transform.py`
3. **Retrieve** → Hybrid search across vector stores → `retrieval/hybrid_retriever.py`
4. **Rerank** → Improve relevance of retrieved docs → `retrieval/reranker.py`
5. **Context Build** → Prepare context with metadata → `rag/context_builder.py`
6. **Generate** → Create response using LLM → `llm/response_generator.py`
7. **Evaluate** → Assess quality (faithfulness, relevancy) → `evaluation/ragas_evaluator.py`
8. **Apply Guardrails** → Validate output safety → `guardrails/`
9. **Collect Feedback** → Store user feedback → `feedback/`
10. **Monitor** → Log traces to Langfuse → `core/langfuse_client.py`

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

### Environment Variables (.env)

**Core Settings:**
- `OPENAI_API_KEY`: Your OpenAI API key for LLM generation
- `OPENAI_MODEL`: Model to use (default: gpt-4o)

**Vector Store:**
- `VECTORSTORE_TYPE`: Choose between `chroma` or `faiss`
- `VECTORSTORE_PATH`: Path to vector store persistence

**Langfuse Monitoring (Optional):**
- `LANGFUSE_PUBLIC_KEY`: Langfuse public key
- `LANGFUSE_SECRET_KEY`: Langfuse secret key
- `LANGFUSE_HOST`: Langfuse backend URL

**Storage:**
- `GCS_BUCKET`: Google Cloud Storage bucket name (optional)
- `DATA_DIR`: Local data directory for uploads

### Config Files (YAML)

- `configs/model_config.yaml`: LLM and embedding model settings
- `configs/pipeline_config.yaml`: Ingestion and processing pipeline configuration
- `configs/retriever_config.yaml`: Retrieval strategy and reranking settings

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

**Health & Status:**
- `GET /`: Health check

**Query & Retrieval:**
- `POST /query`: Query the RAG system with a question
- `POST /chat`: Multi-turn chat with memory
- `GET /search`: Semantic search without generation

**Document Management:**
- `POST /ingest`: Upload and ingest documents
- `GET /documents`: List all indexed documents
- `DELETE /documents/{doc_id}`: Remove a document
- `POST /reindex`: Rebuild vector indices

**Evaluation & Feedback:**
- `POST /evaluate`: Evaluate response quality
- `POST /feedback`: Submit user feedback
- `GET /analytics`: Get usage analytics

### UI Interface

Launch the Streamlit interface:
```bash
streamlit run app/ui/app.py
```

The UI will be available at `http://localhost:8501`

### Example API Usage

```python
import requests

# Query the system
response = requests.post("http://localhost:8000/query",
    json={"query": "What is machine learning?", "top_k": 5})
print(response.json())

# Ingest documents from a file
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/ingest", files=files)
print(response.json())

# Submit feedback
response = requests.post("http://localhost:8000/feedback",
    json={"query_id": "q123", "rating": 5, "comment": "Very helpful"})
print(response.json())
```

## Development

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test suite:
```bash
pytest tests/test_ingestion.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_rag_pipeline.py -v
```

### Code Formatting & Linting

Format with Black:
```bash
black .
```

Sort imports:
```bash
isort .
```

### CLI Scripts

**Ingest Documents:**
```bash
python scripts/ingest_data.py --path ./data/documents
```

**Evaluate System:**
```bash
python scripts/evaluation.py --sample_size 100
```

**Reindex Vectors:**
```bash
python scripts/reindex.py
```

## Deployment

### Docker Deployment

Build and run with Docker Compose:
```bash
docker-compose up --build
```

Services include:
- FastAPI backend (port 8000)
- Streamlit UI (port 8501)
- Vector stores (mounted volumes)

### Cloud Deployment

Deploy to major cloud platforms:
- **AWS**: ECS, Lambda, or EC2
- **GCP**: Cloud Run or App Engine
- **Azure**: Container Instances or App Service

## Monitoring & Observability

### Langfuse Integration

When Langfuse credentials are configured, the system automatically:
- **Logs traces** for every query and generation
- **Tracks latency** across pipeline stages
- **Monitors token usage** and costs
- **Stores conversation history** for analysis
- **Enables debugging** with detailed execution logs

Enable with:
```bash
export LANGFUSE_PUBLIC_KEY=your_key
export LANGFUSE_SECRET_KEY=your_secret
python main.py
```

### Analytics

Access usage metrics via `/analytics` endpoint:
- Query volume and patterns
- Response quality metrics (RAGAS scores)
- User feedback distribution
- System performance statistics

### Logging

Logs are written to:
- Console (development)
- File: `logs/app.log` (production)
- Langfuse (when configured)

## License

MIT License




## Production Features You MUST Include
✅ Logging
- request logs
- retrieval results
- LLM responses

✅ Monitoring
- latency
- token usage
- cost tracking

✅ Evaluation
- faithfulness
- answer relevancy
- context precision

✅ Caching
- query → response cache
- embedding cache

## 🔥 Real-World Flow (End-to-End)
1. Upload PDF → ingestion/
2. Chunk + clean → processors/
3. Generate embeddings → embeddings/
4. Store in → Vertex AI Vector Search
5. Query → retrieval/
6. Context → rag/context_builder.py
7. Response → llm/
8. Serve via → Streamlit or API

