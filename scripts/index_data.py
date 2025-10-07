#!/usr/bin/env python3
"""
embedding and indexing script

generate embeddings for text chunks and upload to pinecone vector database.
"""

import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
from tqdm import tqdm


class EldenRingIndexer:
    def __init__(self):
        self.data_dir = Path("data")
        self.chunks_file = self.data_dir / "text_chunks.json"

        # initialize embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # initialize pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("pinecone_api_key environment variable not set!")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = "elden-ring-wiki-rag"
        self.index = self.pc.Index(self.index_name)

    def load_chunks(self):
        """load text chunks from json file."""
        print("loading text chunks...")
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"loaded {len(chunks)} chunks")
        return chunks

    def generate_embeddings_batch(self, texts, batch_size=32):
        """generate embeddings for a batch of texts."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def prepare_vectors_for_pinecone(self, chunks):
        """prepare chunks and their embeddings for pinecone upload."""
        texts = [chunk["text"] for chunk in chunks]
        print(f"generating embeddings for {len(texts)} chunks...")

        # generate embeddings in batches
        embeddings = self.generate_embeddings_batch(texts)

        # prepare vectors for pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector = {
                "id": chunk["id"],
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk["text"],
                    "url": chunk["metadata"]["url"],
                    "title": chunk["metadata"]["title"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "total_chunks": chunk["metadata"]["total_chunks"],
                    "source": chunk["metadata"]["source"],
                },
            }
            vectors.append(vector)

        return vectors

    def upload_to_pinecone(self, vectors, batch_size=100):
        """upload vectors to pinecone in batches."""
        print(f"uploading {len(vectors)} vectors to pinecone...")

        for i in tqdm(range(0, len(vectors), batch_size)):
            batch = vectors[i : i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"error uploading batch {i//batch_size}: {e}")
                continue

        print("upload complete!")

    def index_all_chunks(self):
        """run the complete indexing pipeline."""
        # load chunks
        chunks = self.load_chunks()

        # prepare vectors
        vectors = self.prepare_vectors_for_pinecone(chunks)

        # upload to pinecone
        self.upload_to_pinecone(vectors)

        # verify upload
        stats = self.index.describe_index_stats()
        print(
            f"indexing complete! index now contains {stats.total_vector_count} vectors"
        )

        return stats


def main():
    try:
        indexer = EldenRingIndexer()
        stats = indexer.index_all_chunks()
        print(f"\n[ok] successfully indexed elden ring wiki data!")
        print(f"[info] total vectors: {stats.total_vector_count}")

    except Exception as e:
        print(f"[error] error during indexing: {e}")
        print("make sure your pinecone_api_key is set in the environment")


if __name__ == "__main__":
    main()
