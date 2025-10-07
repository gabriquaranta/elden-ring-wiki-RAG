#!/usr/bin/env python3
"""
test rag pipeline components

validate individual components of the rag pipeline without api keys.
"""

import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter


def test_data_loading():
    """test loading cleaned data."""
    data_path = Path("data/cleaned_data.json")
    if not data_path.exists():
        print("[fail] cleaned_data.json not found")
        return False

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[ok] loaded {len(data)} pages from cleaned data")
    return True


def test_chunking():
    """test text chunking logic."""
    # load sample data
    data_path = Path("data/cleaned_data.json")
    if not data_path.exists():
        print("[fail] cannot test chunking - no cleaned data")
        return False

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # take first page for testing
    sample_page = data[0]
    text = sample_page["content"]

    # initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # split text
    chunks = text_splitter.split_text(text)

    print(f"[ok] chunked '{sample_page['title']}' into {len(chunks)} chunks")
    print(
        f"   average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars"
    )

    return True


def test_imports():
    """test that all required packages can be imported."""
    try:
        import sentence_transformers

        print("[ok] sentence_transformers imported")
    except ImportError:
        print("[fail] sentence_transformers not available")
        return False

    try:
        import pinecone

        print("[ok] pinecone imported")
    except ImportError:
        print("[fail] pinecone not available")
        return False

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        print("[ok] langchain_google_genai imported")
    except ImportError:
        print("[fail] langchain_google_genai not available")
        return False

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        print("[ok] langchain imported")
    except ImportError:
        print("[fail] langchain not available")
        return False

    return True


def test_api_keys():
    """test that api keys are set and valid."""
    import os

    # test pinecone api key
    pinecone_valid = False
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("pinecone_api_key environment variable not set")

        import pinecone

        pc = pinecone.Pinecone(api_key=api_key)
        index_name = "elden-ring-wiki-rag"

        # check if index exists
        if index_name in pc.list_indexes().names():
            print("[ok] pinecone api key valid and index exists")
            pinecone_valid = True
        else:
            print("[warn] pinecone api key valid but index not found")
            pinecone_valid = True  # key is valid, just no index yet
    except Exception as e:
        print(f"[fail] pinecone api key invalid: {e}")

    # test google api key
    google_valid = False
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("google_api_key environment variable not set")

        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.1,
            max_tokens=10,
        )

        # try a simple api call
        response = llm.invoke("hello")
        print("[ok] google api key valid")
        google_valid = True
    except Exception as e:
        print(f"[fail] google api key invalid: {e}")

    return pinecone_valid and google_valid


def main():
    print("[lab] testing rag pipeline components")
    print("=" * 50)

    tests = [
        ("package imports", test_imports),
        ("data loading", test_data_loading),
        ("text chunking", test_chunking),
        ("api keys", test_api_keys),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n[test] testing {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"[ok] {test_name} passed")
            else:
                print(f"[fail] {test_name} failed")
        except Exception as e:
            print(f"[fail] {test_name} failed with error: {e}")

    print(f"\n[stats] test results: {passed}/{total} passed")

    if passed == total:
        print("[success] all components ready! the full pipeline is ready to run.")
    else:
        print("[warn] some components need attention before proceeding.")


if __name__ == "__main__":
    main()
