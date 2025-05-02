#!/bin/bash

# Update package lists
apt-get update

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Ensure necessary directories exist
mkdir -p assets modules

# Set necessary environment variables (optional, if needed)
export STREAMLIT_SERVER_PORT=8501