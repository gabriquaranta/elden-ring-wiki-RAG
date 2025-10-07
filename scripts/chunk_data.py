#!/usr/bin/env python3
"""
data loading and chunking script

load cleaned wiki data and split into overlapping text chunks for embedding.
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

        # configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 1000 characters per chunk
            chunk_overlap=200,  # 200 character overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_cleaned_data(self):
        """load the cleaned wiki data from json file."""
        print("loading cleaned data...")
        with open(self.cleaned_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"loaded {len(data)} pages")
        return data

    def create_documents(self, data):
        """convert cleaned data to langchain document objects."""
        documents = []
        for page in data:
            # create a document for each page
            doc = Document(
                page_content=page["content"],
                metadata={
                    "url": page["url"],
                    "title": page["title"],
                    "source": "elden_ring_wiki",
                },
            )
            documents.append(doc)

        print(f"created {len(documents)} document objects")
        return documents

    def split_into_chunks(self, documents):
        """split documents into overlapping text chunks."""
        print("splitting documents into chunks...")

        all_chunks = []
        chunk_id = 0

        for doc in documents:
            # split this document into chunks
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

        print(f"created {len(all_chunks)} text chunks")
        return all_chunks

    def save_chunks(self, chunks):
        """save the text chunks to a json file."""
        print(f"saving chunks to {self.chunks_file}...")
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print("chunks saved successfully!")

    def create_chunk_summary(self, chunks):
        """create a summary of the chunking process."""
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

        print(f"summary saved to {summary_file}")

    def process_all_data(self):
        """run the complete data loading and chunking pipeline."""
        # load cleaned data
        data = self.load_cleaned_data()

        # create documents
        documents = self.create_documents(data)

        # split into chunks
        chunks = self.split_into_chunks(documents)

        # save chunks
        self.save_chunks(chunks)

        # create summary
        self.create_chunk_summary(chunks)

        return chunks


def main():
    loader = EldenRingDataLoader()
    chunks = loader.process_all_data()
    print(
        f"\ndata processing complete! created {len(chunks)} chunks ready for embedding."
    )


if __name__ == "__main__":
    main()
