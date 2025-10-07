# Elden Ring Wiki RAG System - Technical Overview

## 🎯 Project Purpose

The Elden Ring Wiki RAG (Retrieval-Augmented Generation) system is an AI-powered question-answering assistant that provides accurate, contextual answers about Elden Ring lore using information from the official Fextralife wiki.

### Key Objectives

- **Accurate Lore Reference**: Provide reliable answers sourced directly from official wiki content
- **Contextual Understanding**: Use advanced AI to understand nuanced questions about complex game lore
- **Source Citation**: Always cite specific wiki pages and provide relevance scores
- **Scalable Architecture**: Handle 90+ wiki pages with efficient vector search and generation

## 🛠️ Technology Stack

### Core AI/ML Frameworks

- **LangChain**: Orchestrates the RAG pipeline, manages document processing and LLM interactions
- **Sentence Transformers**: Generates high-quality embeddings using the `all-MiniLM-L6-v2` model (384-dimensional vectors)
- **Google Gemini 2.5 Flash Preview**: Provides fast, accurate text generation with contextual understanding

### Data Processing & Storage

- **BeautifulSoup4**: Parses and cleans HTML content from wiki pages
- **pandas**: Handles structured data manipulation and analysis
- **Pinecone**: Serverless vector database for efficient similarity search (cosine similarity metric)
- **Python JSON**: Standard library for data serialization and storage

### Web Framework & UI

- **Streamlit**: Creates an interactive web interface for user queries
- **Python pathlib**: Handles file system operations and path management

### Development & Environment

- **Python 3.13**: Core programming language with modern async capabilities
- **pip/requirements.txt**: Dependency management and reproducible environments
- **direnv**: Manages environment variables and API keys securely
- **Git**: Version control and collaborative development

## 🔄 How It Works

### System Architecture

The system follows a classic RAG (Retrieval-Augmented Generation) pattern with four main stages:

```
User Query → Embedding → Retrieval → Generation → Answer
```

### Data Pipeline

#### 1. Data Acquisition (`scripts/scrape.py`)

- **Web Scraping**: Uses `requests` and `BeautifulSoup4` to discover and download wiki pages
- **Rate Limiting**: Implements respectful delays (1 second) between requests
- **Caching**: Stores raw HTML in `data/raw_html/` to avoid re-scraping
- **Discovery**: Automatically finds all relevant wiki pages from the main page and categories

#### 2. Data Processing (`scripts/process.py`)

- **HTML Cleaning**: Removes navigation, ads, scripts, and non-content elements
- **Content Extraction**: Uses multiple strategies to find main article content
- **Text Normalization**: Preserves paragraph structure while cleaning formatting
- **Metadata Extraction**: Captures page titles, URLs, and content summaries

#### 3. Text Chunking (`scripts/chunk_data.py`)

- **Document Splitting**: Divides long text into manageable chunks (1000 characters)
- **Overlap Strategy**: Uses 200-character overlap to maintain context continuity
- **Metadata Preservation**: Maintains source information for each chunk
- **LangChain Integration**: Leverages `RecursiveCharacterTextSplitter` for intelligent splitting

#### 4. Embedding & Indexing (`scripts/index_data.py`)

- **Vector Generation**: Converts text chunks to 384-dimensional embeddings
- **Batch Processing**: Handles large datasets efficiently (32-chunk batches)
- **Pinecone Upload**: Stores vectors with metadata in cloud vector database
- **Index Management**: Creates and manages Pinecone indexes with cosine similarity

### Query Processing

#### 1. Query Embedding (`scripts/query_rag.py`)

- **Input Processing**: Converts user questions to vector embeddings
- **Model Consistency**: Uses same `all-MiniLM-L6-v2` model as document indexing
- **Real-time Encoding**: Processes queries instantly for responsive UX

#### 2. Vector Retrieval (`scripts/query_rag.py`)

- **Similarity Search**: Finds top-5 most relevant chunks using cosine similarity
- **Metadata Filtering**: Includes source URLs, titles, and relevance scores
- **Efficient Lookup**: Pinecone provides sub-second retrieval for large datasets

#### 3. Answer Generation (`scripts/query_rag.py`)

- **Context Injection**: Provides retrieved chunks as context to the LLM
- **Prompt Engineering**: Uses structured prompts for accurate, contextual responses
- **Source Citation**: Requires the model to cite specific sources and relevance scores

## 📊 Performance Characteristics

### Data Scale

- **Content Volume**: 93 wiki pages, 5M+ characters of processed text
- **Vector Database**: ~1,200 text chunks stored as 384-dimensional vectors
- **Index Size**: Serverless Pinecone deployment with automatic scaling

### Resource Efficiency

- **Memory Usage**: Lightweight embedding model fits in standard RAM
- **API Costs**: Optimized for cost-effective LLM usage
- **Storage**: Minimal disk usage with efficient JSON storage formats

## 🎯 Key Components Deep Dive

### Embedding Model: `all-MiniLM-L6-v2`

- **Architecture**: 6-layer transformer with mean pooling
- **Dimensions**: 384-dimensional vectors balancing quality and speed
- **Training**: Fine-tuned on diverse text corpora for semantic understanding
- **Performance**: Excellent semantic similarity capture for lore-related queries

### Vector Database: Pinecone

- **Similarity Metric**: Cosine similarity for semantic relevance
- **Cloud Infrastructure**: AWS us-east-1 serverless deployment
- **Scalability**: Automatic scaling based on query volume
- **Metadata Support**: Rich metadata storage for source attribution

### LLM: Google Gemini 2.5 Flash Preview

- **Model Size**: Optimized for speed and efficiency
- **Context Window**: Sufficient for retrieved chunks + generation
- **Fine-tuning**: Pre-trained on diverse web text for general knowledge
- **API Integration**: Seamless LangChain integration for RAG workflows

### Text Processing Pipeline

- **Chunk Strategy**: 1000-char chunks with 200-char overlap
- **Separator Priority**: `\n\n` > `\n` > `. ` > ` ` > `` (preserves structure)
- **Content Filtering**: Removes boilerplate while preserving lore content
- **Quality Assurance**: Multiple fallback strategies for content extraction

## 🚀 Use Cases & Applications

### For Players

- **Lore Research**: Deep dive into character backstories and world history
- **Mechanic Clarification**: Understand complex game systems and interactions
- **Story Analysis**: Connect plot points across the vast game narrative

### For Content Creators

- **Fact Checking**: Verify lore accuracy for guides and videos
- **Research Support**: Quickly gather information for content creation
- **Educational Content**: Build accurate lore explanations and theories

### For Developers

- **RAG Implementation**: Reference architecture for similar projects
- **AI Integration**: Example of combining embeddings, vector search, and LLMs
- **Data Processing**: Template for web scraping and text processing pipelines

## 🔧 Development Workflow

### Local Development

1. **Environment Setup**: `python -m venv .env && source .env/bin/activate`
2. **Dependencies**: `pip install -r requirements.txt`
3. **API Keys**: Configure Pinecone and Google Gemini in `.envrc`
4. **Data Pipeline**: Run scraping → processing → chunking → indexing
5. **Testing**: `python scripts/test_pipeline.py` validates all components

### Deployment Considerations

- **API Key Management**: Secure environment variable handling
- **Scalability**: Pinecone handles increased query loads automatically
- **Cost Optimization**: Efficient chunking and retrieval strategies
- **Monitoring**: Track query performance and accuracy metrics

## 🎖️ Quality Assurance

### Testing Strategy

- **Component Validation**: Individual script testing with `test_pipeline.py`
- **Integration Testing**: End-to-end pipeline verification
- **API Validation**: Real-time checking of Pinecone and Gemini connectivity
- **Data Integrity**: Checksums and validation of processed content

### Accuracy Measures

- **Source Verification**: All answers cite specific wiki pages
- **Relevance Scoring**: Cosine similarity scores for retrieval quality
- **Context Completeness**: Multiple chunks provide comprehensive answers
- **Factual Consistency**: LLM generates based on verified wiki content

## 🔮 Future Enhancements

### Potential Improvements

- **GraphRAG**: Add entity relationship mapping for complex queries
- **Multi-modal**: Include images and maps in responses
- **Conversational Memory**: Maintain context across multiple queries
- **Evaluation Framework**: Automated testing against ground truth answers

### Scalability Considerations

- **Larger Datasets**: Support for additional wiki sources and expansions
- **Query Optimization**: Advanced retrieval strategies and re-ranking
- **Caching Layer**: Reduce API costs for frequently asked questions
- **Multi-language**: Support for international wiki translations


