#!/usr/bin/env python3
"""
Data Loading and Chunking Script

Load cleaned wiki data and split into overlapping text chunks for embedding.
"""

import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd


class EldenRingDataLoader:
    def __init__(self):
        self.data_dir = Path("data")
        self.cleaned_data_file = self.data_dir / "cleaned_data.json"
        self.chunks_file = self.data_dir / "text_chunks.json"

        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 1000 characters per chunk
            chunk_overlap=200,  # 200 character overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_cleaned_data(self):
        """Load the cleaned wiki data from JSON file."""
        print("Loading cleaned data...")
        with open(self.cleaned_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} pages")
        return data

    def create_documents(self, data):
        """Convert cleaned data to LangChain Document objects."""
        documents = []
        for page in data:
            # Create a document for each page
            doc = Document(
                page_content=page["content"],
                metadata={
                    "url": page["url"],
                    "title": page["title"],
                    "source": "elden_ring_wiki",
                },
            )
            documents.append(doc)

        print(f"Created {len(documents)} document objects")
        return documents

    def split_into_chunks(self, documents):
        """Split documents into overlapping text chunks."""
        print("Splitting documents into chunks...")

        all_chunks = []
        chunk_id = 0

        for doc in documents:
            # Split this document into chunks
            chunks = self.text_splitter.split_text(doc.page_content)

            for i, chunk_text in enumerate(chunks):
                chunk_data = {
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                }
                all_chunks.append(chunk_data)
                chunk_id += 1

        print(f"Created {len(all_chunks)} text chunks")
        return all_chunks

    def save_chunks(self, chunks):
        """Save the text chunks to a JSON file."""
        print(f"Saving chunks to {self.chunks_file}...")
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print("Chunks saved successfully!")

    def create_chunk_summary(self, chunks):
        """Create a summary of the chunking process."""
        summary = {
            "total_chunks": len(chunks),
            "total_characters": sum(len(chunk["text"]) for chunk in chunks),
            "average_chunk_length": (
                sum(len(chunk["text"]) for chunk in chunks) / len(chunks)
                if chunks
                else 0
            ),
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "sample_chunks": chunks[:3] if len(chunks) >= 3 else chunks,
        }

        summary_file = self.data_dir / "chunking_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Summary saved to {summary_file}")

    def process_all_data(self):
        """Run the complete data loading and chunking pipeline."""
        # Load cleaned data
        data = self.load_cleaned_data()

        # Create documents
        documents = self.create_documents(data)

        # Split into chunks
        chunks = self.split_into_chunks(documents)

        # Save chunks
        self.save_chunks(chunks)

        # Create summary
        self.create_chunk_summary(chunks)

        return chunks


def main():
    loader = EldenRingDataLoader()
    chunks = loader.process_all_data()
    print(
        f"\nData processing complete! Created {len(chunks)} chunks ready for embedding."
    )


if __name__ == "__main__":
    main()
