#!/usr/bin/env python3
"""
Embedding and Indexing Script

Generate embeddings for text chunks and upload to Pinecone vector database.
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

        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Pinecone
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set!")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = "elden-ring-wiki-rag"
        self.index = self.pc.Index(self.index_name)

    def load_chunks(self):
        """Load text chunks from JSON file."""
        print("Loading text chunks...")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
        return chunks

    def generate_embeddings_batch(self, texts, batch_size=32):
        """Generate embeddings for a batch of texts."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def prepare_vectors_for_pinecone(self, chunks):
        """Prepare chunks and their embeddings for Pinecone upload."""
        texts = [chunk['text'] for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")

        # Generate embeddings in batches
        embeddings = self.generate_embeddings_batch(texts)

        # Prepare vectors for Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector = {
                'id': chunk['id'],
                'values': embedding.tolist(),
                'metadata': {
                    'text': chunk['text'],
                    'url': chunk['metadata']['url'],
                    'title': chunk['metadata']['title'],
                    'chunk_index': chunk['metadata']['chunk_index'],
                    'total_chunks': chunk['metadata']['total_chunks'],
                    'source': chunk['metadata']['source']
                }
            }
            vectors.append(vector)

        return vectors

    def upload_to_pinecone(self, vectors, batch_size=100):
        """Upload vectors to Pinecone in batches."""
        print(f"Uploading {len(vectors)} vectors to Pinecone...")

        for i in tqdm(range(0, len(vectors), batch_size)):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error uploading batch {i//batch_size}: {e}")
                continue

        print("Upload complete!")

    def index_all_chunks(self):
        """Run the complete indexing pipeline."""
        # Load chunks
        chunks = self.load_chunks()

        # Prepare vectors
        vectors = self.prepare_vectors_for_pinecone(chunks)

        # Upload to Pinecone
        self.upload_to_pinecone(vectors)

        # Verify upload
        stats = self.index.describe_index_stats()
        print(f"Indexing complete! Index now contains {stats.total_vector_count} vectors")

        return stats

def main():
    try:
        indexer = EldenRingIndexer()
        stats = indexer.index_all_chunks()
        print(f"\n✅ Successfully indexed Elden Ring wiki data!")
        print(f"📊 Total vectors: {stats.total_vector_count}")

    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        print("Make sure your PINECONE_API_KEY is set in the environment")

if __name__ == "__main__":
    main()