#!/bin/bash
set -e

# Update pip
pip install --upgrade pip

# Install system dependencies for building FAISS and numpy
apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran

# Install Python dependencies (force source build for faiss-cpu to handle compatibility)
pip install -r requirements.txt --no-binary faiss-cpu

# Download NLTK stopwords (your code requires it)
python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Clean up to reduce image size
apt-get clean && rm -rf /var/lib/apt/lists/*
