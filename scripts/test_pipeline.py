#!/usr/bin/env python3
"""
Test RAG Pipeline Components

Validate individual components of the RAG pipeline without API keys.
"""

import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter


def test_data_loading():
    """Test loading cleaned data."""
    data_path = Path("data/cleaned_data.json")
    if not data_path.exists():
        print("❌ cleaned_data.json not found")
        return False

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} pages from cleaned data")
    return True


def test_chunking():
    """Test text chunking logic."""
    # Load sample data
    data_path = Path("data/cleaned_data.json")
    if not data_path.exists():
        print("❌ Cannot test chunking - no cleaned data")
        return False

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Take first page for testing
    sample_page = data[0]
    text = sample_page["content"]

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split text
    chunks = text_splitter.split_text(text)

    print(f"✅ Chunked '{sample_page['title']}' into {len(chunks)} chunks")
    print(
        f"   Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars"
    )

    return True


def test_imports():
    """Test that all required packages can be imported."""
    try:
        import sentence_transformers

        print("✅ sentence_transformers imported")
    except ImportError:
        print("❌ sentence_transformers not available")
        return False

    try:
        import pinecone

        print("✅ pinecone imported")
    except ImportError:
        print("❌ pinecone not available")
        return False

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        print("✅ langchain_google_genai imported")
    except ImportError:
        print("❌ langchain_google_genai not available")
        return False

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        print("✅ langchain imported")
    except ImportError:
        print("❌ langchain not available")
        return False

    return True


def test_api_keys():
    """Test that API keys are set and valid."""
    import os

    # Test Pinecone API key
    pinecone_valid = False
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        import pinecone

        pc = pinecone.Pinecone(api_key=api_key)
        index_name = "elden-ring-wiki-rag"

        # Check if index exists
        if index_name in pc.list_indexes().names():
            print("✅ Pinecone API key valid and index exists")
            pinecone_valid = True
        else:
            print("⚠️  Pinecone API key valid but index not found")
            pinecone_valid = True  # Key is valid, just no index yet
    except Exception as e:
        print(f"❌ Pinecone API key invalid: {e}")

    # Test Google API key
    google_valid = False
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.1,
            max_tokens=10,
        )

        # Try a simple API call
        response = llm.invoke("Hello")
        print("✅ Google API key valid")
        google_valid = True
    except Exception as e:
        print(f"❌ Google API key invalid: {e}")

    return pinecone_valid and google_valid


def main():
    print("🧪 Testing RAG Pipeline Components")
    print("=" * 50)

    tests = [
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Text Chunking", test_chunking),
        ("API Keys", test_api_keys),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All components ready! The full pipeline is ready to run.")
    else:
        print("⚠️  Some components need attention before proceeding.")


if __name__ == "__main__":
    main()
