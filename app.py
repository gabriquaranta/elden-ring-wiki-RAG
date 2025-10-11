#!/usr/bin/env python3
"""
elden ring lore assistant - streamlit web app

a simple web interface for querying the elden ring wiki rag system.
"""

import streamlit as st
import os
from pathlib import Path

# add the scripts directory to the path so we can import our modules
import sys

sys.path.append(str(Path(__file__).parent / "scripts"))

# do not initialize the RAG system at import time (it may perform network IO)
rag_system = None


def main():
    global rag_system
    st.set_page_config(
        page_title="🗡️ Elden Ring Lore Assistant",
        page_icon="⚔️",
        layout="wide",
    )

    st.title("🗡️ Elden Ring Lore Assistant")
    st.markdown("*powered by RAG and Google Gemini*")

    # initialize conversation history in session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
        this ai assistant answers questions about elden ring lore using information from the official wiki.

        **features:**
        - searches through 90+ wiki pages
        - provides accurate, contextual answers
        - cites sources from the wiki
        """
        )

        st.header("🔧 Setup Status")
        # initialize the RAG system lazily so importing this module doesn't block
        if rag_system is None:
            try:
                from query_rag import EldenRingRAG

                rag_system = EldenRingRAG()
                st.success("✅ RAG system ready!")
            except Exception as e:
                st.error("❌ RAG system not initialized. Check API keys.")
                st.write(f"initialization error: {e}")
                rag_system = None
        else:
            st.success("✅ RAG system ready!")

        st.header("📚 Sample Questions")
        sample_questions = [
            "who is queen marika?",
            "what are the different types of damage?",
            "how do i reach the mountaintops of the giants?",
            "what are the requirements to become elden lord?",
        ]

        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.question = q

    # main content
    if rag_system is None:
        st.error(
            "🚫 The RAG system couldn't be initialized. Please check that your API keys are set up correctly in `.envrc`."
        )
        st.code(
            """
# add these lines to your .envrc file:
pinecone_api_key=your_pinecone_key_here
google_api_key=your_google_key_here
        """
        )
        return

    # question input
    # show conversation history
    if st.session_state.history:
        st.markdown("### Conversation")
        for turn in st.session_state.history:
            st.markdown(f"**User:** {turn.get('user','')} ")
            st.markdown(f"**Assistant:** {turn.get('assistant','')} ")
            st.divider()

    question = st.text_input(
        "ask a question about elden ring lore:",
        value=st.session_state.get("question", ""),
        placeholder="e.g., who is the first demigod you encounter?",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔁 Clear conversation"):
            st.session_state.history = []
            st.session_state.question = ""
            st.experimental_rerun()

    if st.button("🔍 Search Lore", type="primary") and question.strip():
        with st.spinner("searching ancient tomes..."):
            try:
                # pass conversation history to the rag system and receive updated history
                answer, chunks, history = rag_system.answer_question(
                    question.strip(), history=st.session_state.history
                )

                # update session history
                st.session_state.history = history

                # display answer
                st.success("answer found!")
                st.markdown("### 💬 Answer")
                st.write(answer)

                # display sources
                with st.expander("📄 Sources", expanded=False):
                    st.markdown(f"**found {len(chunks)} relevant sources:**")
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**source {i}** (relevance: {chunk['score']:.3f})")
                        st.markdown(f"*{chunk['title']}*")
                        st.markdown(f"[view on wiki]({chunk['url']})")
                        st.text_area(
                            f"content preview:",
                            (
                                chunk["text"][:500] + "..."
                                if len(chunk["text"]) > 500
                                else chunk["text"]
                            ),
                            height=100,
                            key=f"chunk_{i}",
                        )
                        st.divider()

            except Exception as e:
                st.error(f"❌ Error during search: {e}")
                st.info("this might be due to api key issues or network problems.")

    # footer
    st.markdown("---")
    st.markdown("*built with streamlit, langchain, pinecone, and google gemini*")


if __name__ == "__main__":
    main()
