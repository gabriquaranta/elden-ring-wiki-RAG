#!/usr/bin/env python3
"""
Elden Ring Lore Assistant - Streamlit Web App

A simple web interface for querying the Elden Ring wiki RAG system.
"""

import streamlit as st
import os
from pathlib import Path

# Add the scripts directory to the path so we can import our modules
import sys
sys.path.append(str(Path(__file__).parent / "scripts"))

# Import our RAG system (this will fail if API keys aren't set)
try:
    from query_rag import EldenRingRAG
    rag_system = EldenRingRAG()
except Exception as e:
    st.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

def main():
    st.set_page_config(
        page_title="🗡️ Elden Ring Lore Assistant",
        page_icon="⚔️",
        layout="wide"
    )

    st.title("🗡️ Elden Ring Lore Assistant")
    st.markdown("*Powered by RAG and Google Gemini*")

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI assistant answers questions about Elden Ring lore using information from the official wiki.

        **Features:**
        - Searches through 90+ wiki pages
        - Provides accurate, contextual answers
        - Cites sources from the wiki
        """)

        st.header("🔧 Setup Status")
        if rag_system is None:
            st.error("❌ RAG system not initialized. Check API keys.")
        else:
            st.success("✅ RAG system ready!")

        st.header("📚 Sample Questions")
        sample_questions = [
            "Who is Queen Marika?",
            "What are the different types of damage?",
            "How do I reach the Mountaintops of the Giants?",
            "What are the requirements to become Elden Lord?"
        ]

        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.question = q

    # Main content
    if rag_system is None:
        st.error("🚫 The RAG system couldn't be initialized. Please check that your API keys are set up correctly in `.envrc`.")
        st.code("""
# Add these lines to your .envrc file:
PINECONE_API_KEY=your_pinecone_key_here
GOOGLE_API_KEY=your_google_key_here
        """)
        return

    # Question input
    question = st.text_input(
        "Ask a question about Elden Ring lore:",
        value=st.session_state.get('question', ''),
        placeholder="e.g., Who is the first demigod you encounter?"
    )

    if st.button("🔍 Search Lore", type="primary") and question.strip():
        with st.spinner("Searching ancient tomes..."):
            try:
                answer, chunks = rag_system.answer_question(question.strip())

                # Display answer
                st.success("Answer found!")
                st.markdown("### 💬 Answer")
                st.write(answer)

                # Display sources
                with st.expander("📄 Sources", expanded=False):
                    st.markdown(f"**Found {len(chunks)} relevant sources:**")
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Source {i}** (Relevance: {chunk['score']:.3f})")
                        st.markdown(f"*{chunk['title']}*")
                        st.markdown(f"[View on Wiki]({chunk['url']})")
                        st.text_area(
                            f"Content preview:",
                            chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                            height=100,
                            key=f"chunk_{i}"
                        )
                        st.divider()

            except Exception as e:
                st.error(f"❌ Error during search: {e}")
                st.info("This might be due to API key issues or network problems.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, LangChain, Pinecone, and Google Gemini*")

if __name__ == "__main__":
    main()