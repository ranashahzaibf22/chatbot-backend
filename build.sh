#!/bin/bash
set -e

# Update pip
pip install --upgrade pip

# Set CARGO_HOME to a writable directory to avoid read-only file system errors
export CARGO_HOME=/tmp/cargo

# Install system dependencies for building FAISS and Rust-based packages
apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    pkg-config \
    libssl-dev \
    rustc \
    cargo

# Install Python dependencies
pip install -r requirements.txt --no-binary :all: faiss-cpu

# Optional: Clean up to reduce image size
apt-get clean && rm -rf /var/lib/apt/lists/*
