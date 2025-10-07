#!/usr/bin/env python3
"""
RAG Query Chain

Build and run the RAG pipeline for answering Elden Ring lore questions.
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


class EldenRingRAG:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set!")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = "elden-ring-wiki-rag"
        self.index = self.pc.Index(self.index_name)

        # Initialize Gemini LLM
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set!")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.1,
            max_tokens=1024,
        )

        # Set up the RAG prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert on Elden Ring lore. Use the following context from the Elden Ring wiki to answer the user's question accurately and helpfully.

Context from Elden Ring Wiki:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- Be accurate and detailed but concise
- If the context doesn't contain enough information, say so clearly
- Include relevant quotes from the context when helpful
- Maintain an engaging, helpful tone

Answer:""",
        )

    def retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieve the most relevant text chunks for a query."""
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query])[0]

        # Search Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(), top_k=top_k, include_metadata=True
        )

        # Extract text chunks from results
        chunks = []
        for match in results["matches"]:
            chunk_text = match["metadata"]["text"]
            chunks.append(
                {
                    "text": chunk_text,
                    "score": match["score"],
                    "title": match["metadata"]["title"],
                    "url": match["metadata"]["url"],
                }
            )

        return chunks

    def format_context(self, chunks):
        """Format retrieved chunks into context string."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}] {chunk['title']}\n{chunk['text']}")

        return "\n\n".join(context_parts)

    def answer_question(self, question):
        """Answer a question using the RAG pipeline."""
        print(f"🔍 Searching for: {question}")

        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(question)
        print(f"📚 Found {len(chunks)} relevant chunks")

        if not chunks:
            return "I couldn't find any relevant information in the Elden Ring wiki for this question."

        # Format context
        context = self.format_context(chunks)

        # Create the chain
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Generate answer
        print("🤖 Generating answer...")
        answer = chain.invoke(question)

        return answer, chunks


def main():
    # Example usage
    rag = EldenRingRAG()

    # Test questions
    test_questions = [
        "Who is Queen Marika?",
        "What are the different types of damage in Elden Ring?",
        "How do I get to the Mountaintops of the Giants?",
        "What are the requirements for becoming the Elden Lord?",
    ]

    print("🗡️ Elden Ring Lore Assistant")
    print("=" * 50)

    for question in test_questions:
        print(f"\n❓ {question}")
        try:
            answer, chunks = rag.answer_question(question)
            print(f"💬 {answer}")
            print(f"📄 Sources: {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 50)


if __name__ == "__main__":
    main()
