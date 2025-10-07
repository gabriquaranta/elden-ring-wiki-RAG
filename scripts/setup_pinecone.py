#!/usr/bin/env python3
"""
pinecone vector database setup

initialize pinecone client and create index for elden ring rag.
"""

import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
import time


def setup_pinecone():
    """set up pinecone index for the rag system."""

    # initialize pinecone client
    # try to read pinecone key from ./api-keys.txt (format: pinecone=<apikey>), fall back to env var
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
        print("error: pinecone_api_key environment variable not set!")
        print("please set your pinecone api key:")
        print("export pinecone_api_key='your-api-key-here'")
        return None

    pc = Pinecone(api_key=api_key)

    # define index name and configuration
    index_name = "elden-ring-wiki-rag"
    dimension = 384  # dimension for 'all-minilm-l6-v2' embeddings
    metric = "cosine"

    # check if index already exists
    if index_name in pc.list_indexes().names():
        print(f"index '{index_name}' already exists.")
        index = pc.Index(index_name)
        print(f"index stats: {index.describe_index_stats()}")
        return index

    # create new index
    print(f"creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # wait for index to be ready
    print("waiting for index to be ready...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    print(f"index '{index_name}' created successfully!")
    index = pc.Index(index_name)
    return index


def test_pinecone_connection():
    """test the pinecone connection and index."""
    try:
        index = setup_pinecone()
        if index:
            # test with a simple query
            test_vector = [0.1] * 384  # dummy vector for testing
            results = index.query(vector=test_vector, top_k=1, include_metadata=True)
            print("pinecone connection test successful!")
            print(f"index contains {results.total_vector_count} vectors")
            return True
    except Exception as e:
        print(f"pinecone connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("setting up pinecone for elden ring wiki rag...")
    test_pinecone_connection()
