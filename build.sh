#!/bin/bash
set -e

# Update pip
pip install --upgrade pip

# Install system dependencies for building FAISS
apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran

# Install Python dependencies
pip install -r requirements.txt --no-binary :all: faiss-cpu

# Optional: Clean up to reduce image size
apt-get clean && rm -rf /var/lib/apt/lists/*
