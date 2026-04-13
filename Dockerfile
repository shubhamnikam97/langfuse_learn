# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/

# Create data directory
RUN mkdir -p data logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]