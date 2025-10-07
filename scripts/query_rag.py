#!/usr/bin/env python3
"""
rag query chain

build and run the rag pipeline for answering elden ring lore questions.
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
        # initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # initialize pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("pinecone_api_key environment variable not set!")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = "elden-ring-wiki-rag"
        self.index = self.pc.Index(self.index_name)

        # initialize gemini llm
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("google_api_key environment variable not set!")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.1,
            max_tokens=1024,
        )

        # set up the rag prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""you are an expert on elden ring lore. use the following context from the elden ring wiki to answer the user's question accurately and helpfully.

context from elden ring wiki:
{context}

question: {question}

instructions:
- answer based primarily on the provided context
- be accurate and detailed but concise
- if the context doesn't contain enough information, say so clearly
- include relevant quotes from the context when helpful
- maintain an engaging, helpful tone

answer:""",
        )

    def retrieve_relevant_chunks(self, query, top_k=5):
        """retrieve the most relevant text chunks for a query."""
        # generate embedding for the query
        query_embedding = self.embedding_model.encode([query])[0]

        # search pinecone
        results = self.index.query(
            vector=query_embedding.tolist(), top_k=top_k, include_metadata=True
        )

        # extract text chunks from results
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
        """format retrieved chunks into context string."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[source {i}] {chunk['title']}\n{chunk['text']}")

        return "\n\n".join(context_parts)

    def answer_question(self, question):
        """answer a question using the rag pipeline."""
        print(f"[search] searching for: {question}")

        # retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(question)
        print(f"[books] found {len(chunks)} relevant chunks")

        if not chunks:
            return "i couldn't find any relevant information in the elden ring wiki for this question."

        # format context
        context = self.format_context(chunks)

        # create the chain
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        # generate answer
        print("[ai] generating answer...")
        answer = chain.invoke(question)

        return answer, chunks


def main():
    # example usage
    rag = EldenRingRAG()

    # test questions
    test_questions = [
        "who is queen marika?",
        "what are the different types of damage in elden ring?",
        "how do i get to the mountaintops of the giants?",
        "what are the requirements for becoming the elden lord?",
    ]

    print("[sword] elden ring lore assistant")
    print("=" * 50)

    for question in test_questions:
        print(f"\n[q] {question}")
        try:
            answer, chunks = rag.answer_question(question)
            print(f"[answer] {answer}")
            print(f"[pages] sources: {len(chunks)} chunks")
        except Exception as e:
            print(f"[error] error: {e}")

        print("-" * 50)


if __name__ == "__main__":
    main()
