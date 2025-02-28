# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Copy source code and data (these layers will be rebuilt when files change)
COPY src/ ./src/
COPY data/ ./data/

# Expose port for Gradio interface
EXPOSE 7860

# Run the pet breed classifier
CMD ["python", "src/pet_classifier.py"]