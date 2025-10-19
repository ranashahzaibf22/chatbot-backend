# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data to a persistent location
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data', quiet=True)"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p faiss_index model_cache uploads

# Expose port (Railway will set PORT env variable)
EXPOSE 8000

# Command to run the application
# Railway sets PORT env variable, default to 8000 if not set
CMD uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000}
