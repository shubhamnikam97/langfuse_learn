# Use lightweight Python image
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync

# Copy full project
COPY . .

# Expose fastapi port
EXPOSE 8000

# Start Fastapi server
CMD ["uv", "run","uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]