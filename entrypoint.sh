#!/bin/bash
set -e

echo "Checking model integrity..."
python3 download_models.py

# Start the FastAPI Server
echo "Starting Inference Server..."
uvicorn server:app --host 0.0.0.0 --port 8000