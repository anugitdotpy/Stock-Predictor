# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y pciutils curl && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create the nltk_data folder and set permissions
RUN mkdir -p /nltk_data && chmod -R 777 /nltk_data
# Create the .ollama folder and set permissions so Ollama can generate its key
RUN mkdir -p /.ollama && chmod -R 777 /.ollama

# Create a writable cache directory for Hugging Face
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache
ENV HF_HOME=/app/.cache

# Set environment variable for NLTK data
ENV NLTK_DATA=/nltk_data

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install sentencepiece required by Helsinki-NLP tokenizers
RUN pip install --no-cache-dir sentencepiece

# Copy the rest of your application code
COPY . .

# Expose port 5060 (or whichever port your FastAPI app uses)
EXPOSE 5060

# Command to run the application with uvicorn (using a single worker to avoid port conflicts)
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "5060", "--workers", "1"]
