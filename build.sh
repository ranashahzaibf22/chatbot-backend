#!/bin/bash
set -e

# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install system dependencies for faiss-cpu and numpy
apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran

# Install Python dependencies with verbose output for debugging
pip install -r requirements.txt --no-binary faiss-cpu -v

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Clean up to reduce image size
apt-get clean && rm -rf /var/lib/apt/lists/*
