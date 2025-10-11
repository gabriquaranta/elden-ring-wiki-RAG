#!/usr/bin/env python3
"""
rag query chain

build and run the rag pipeline for answering elden ring lore questions.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

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

        # set up the rag prompt template (now supports conversation history)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "history", "question"],
            template="""you are an expert on elden ring lore. use the following context from the elden ring wiki to answer the user's question accurately and helpfully.

conversation history (most recent last):
{history}

context from elden ring wiki:
{context}

question: {question}

instructions:
- answer based primarily on the provided context and relevant parts of the conversation history
- be accurate and detailed but concise
- if the context doesn't contain enough information, say so clearly
- include relevant quotes from the context when helpful
- when the user asks a follow-up question, use the conversation history to resolve references
- maintain an engaging, helpful tone

answer:""",
        )

        # how many past turns to keep in the history when building the prompt
        self.history_max_turns = 6

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

    def format_history(self, history: Optional[List[Dict[str, str]]]) -> str:
        """format a list of turns into a string for the prompt.

        history is a list of dicts with keys: 'user' and 'assistant'.
        the function keeps only the last `history_max_turns` turns.
        """
        if not history:
            return "(no prior conversation)"

        # keep only the last N turns
        trimmed = history[-self.history_max_turns :]
        parts = []
        for turn in trimmed:
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            parts.append(f"User: {user_text}\nAssistant: {assistant_text}")

        return "\n\n".join(parts)

    def answer_question(
        self, question: str, history: Optional[List[Dict[str, str]]] = None
    ):
        """answer a question using the rag pipeline."""
        print(f"[search] searching for: {question}")

        # retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(question)
        print(f"[books] found {len(chunks)} relevant chunks")

        if not chunks:
            answer_text = "i couldn't find any relevant information in the elden ring wiki for this question."
            if history is None:
                history = []
            history.append({"user": question, "assistant": answer_text})
            return answer_text, chunks, history

        # format context and conversation history
        context = self.format_context(chunks)
        history_str = self.format_history(history)

        # create the chain
        chain = (
            {
                "context": lambda x: context,
                "history": lambda x: history_str,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        # generate answer
        print("[ai] generating answer...")
        answer = chain.invoke(question)

        # ensure history list exists and append this turn
        if history is None:
            history = []

        history.append({"user": question, "assistant": answer})

        return answer, chunks, history


def main():
    rag = EldenRingRAG()

    print("[sword] elden ring lore assistant (multi-turn)")
    print("type 'exit' or 'quit' to leave, 'clear' to reset conversation history")
    print("=" * 60)

    history: List[Dict[str, str]] = []

    while True:
        try:
            question = input("\n[q] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nexiting...")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("goodbye")
            break

        if question.lower() == "clear":
            history = []
            print("conversation history cleared")
            continue

        try:
            answer, chunks, history = rag.answer_question(question, history=history)
            print(f"\n[answer] {answer}\n")
            print(f"[pages] sources: {len(chunks)} chunks")
        except Exception as e:
            print(f"[error] error: {e}")

        print("-" * 60)


if __name__ == "__main__":
    main()
