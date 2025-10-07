#!/usr/bin/env python3
"""
Pinecone Vector Database Setup

Initialize Pinecone client and create index for Elden Ring RAG.
"""

import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
import time


def setup_pinecone():
    """Set up Pinecone index for the RAG system."""

    # initialize Pinecone client
    # try to read pinecone key from ./api-keys.txt (format: PINECONE=<apikey>), fall back to env var
    api_key = None
    try:
        keys_path = os.path.join(os.getcwd(), "api-keys.txt")
        if os.path.exists(keys_path):
            with open(keys_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.upper().startswith("PINECONE="):
                        api_key = line.split("=", 1)[1].strip()
                        break
                    # if a key/value line exists, try to parse it
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip().upper() == "PINECONE":
                            api_key = v.strip()
                            break
        if not api_key:
            api_key = os.getenv("PINECONE_API_KEY")
    except Exception as e:
        print(f"warning: failed to read api-keys.txt: {e}")
        api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set!")
        print("Please set your Pinecone API key:")
        print("export PINECONE_API_KEY='your-api-key-here'")
        return None

    pc = Pinecone(api_key=api_key)

    # Define index name and configuration
    index_name = "elden-ring-wiki-rag"
    dimension = 384  # Dimension for 'all-MiniLM-L6-v2' embeddings
    metric = "cosine"

    # Check if index already exists
    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists.")
        index = pc.Index(index_name)
        print(f"Index stats: {index.describe_index_stats()}")
        return index

    # Create new index
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Wait for index to be ready
    print("Waiting for index to be ready...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    print(f"Index '{index_name}' created successfully!")
    index = pc.Index(index_name)
    return index


def test_pinecone_connection():
    """Test the Pinecone connection and index."""
    try:
        index = setup_pinecone()
        if index:
            # Test with a simple query
            test_vector = [0.1] * 384  # Dummy vector for testing
            results = index.query(vector=test_vector, top_k=1, include_metadata=True)
            print("Pinecone connection test successful!")
            print(f"Index contains {results.total_vector_count} vectors")
            return True
    except Exception as e:
        print(f"Pinecone connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("Setting up Pinecone for Elden Ring Wiki RAG...")
    test_pinecone_connection()
