# Elden Ring Wiki RAG Project Roadmap

### **Phase 1: Foundational Setup & Data Acquisition**

_The goal of this phase is to establish a solid project structure and gather high-quality, clean data from the wiki._

- **[X] Project Initialization**

  - **Action:** Create a new Git repository for your project.
  - **Action:** Set up a Python virtual environment (`venv` or `conda`).
  - **Tech:** Git, Python

- **[X] Initial Dependency Installation**

  - **Action:** Install core libraries for scraping and data handling.
  - **Tech:** `pip install requests beautifulsoup4 pandas notebook`

- **[X] Develop Scraping Script**

  - **Action:** Write a Python script to scrape the text content from all relevant pages of the Elden Ring Fextralife wiki.
  - **Action:** Implement robust logic to handle potential issues like dynamic JavaScript-loaded content (if necessary).
  - **Tech:** `requests`, `BeautifulSoup4`, `Selenium` (if needed)

- **[X] Cache Raw Data**

  - **Action:** Save the raw, unprocessed HTML of every scraped page into a local directory (`data/raw_html/`). This prevents needing to re-scrape constantly.

- **[X] Develop Cleaning & Structuring Script**
  - **Action:** Write a script that reads the raw HTML files, extracts the primary text content, and removes irrelevant HTML tags, navigation bars, and ads.
  - **Action:** Save the cleaned content in a structured format. A single JSON file containing a list of objects (each with `url`, `title`, and `content` keys) is ideal.
  - **Tech:** `BeautifulSoup4`, `pandas` or Python's `json` library

### **Phase 2: Core RAG Pipeline Implementation**

_With clean data in hand, this phase focuses on building the main retrieval and generation engine._

- **[ ] Install Core ML/LLM Libraries**

  - **Action:** Install the necessary frameworks and libraries for the RAG pipeline.
  - **Tech:** `pip install langchain openai sentence-transformers` (or `google-generativeai`, etc.)

- **[ ] Install and Set Up Vector Database**

  - **Action:** Choose a vector database, sign up for a free-tier account, and get your API key.
  - **Action:** Install the specific Python client for your chosen database.
  - **Tech:** `pip install pinecone-client`, `qdrant-client`, or `weaviate-client`

- **[ ] Implement Data Loading and Chunking**

  - **Action:** Write the script to load your cleaned JSON data.
  - **Action:** Use your chosen framework (`LangChain` or `LlamaIndex`) to split the documents into smaller, overlapping text chunks.

- **[ ] Implement Embedding and Indexing**

  - **Action:** Write the script that performs the following:
    1.  Initializes the embedding model (e.g., from `sentence-transformers`).
    2.  Connects to your vector database.
    3.  Iterates through each text chunk, generates a vector embedding, and uploads the chunk and its embedding to the database. This is a one-time setup process.

- **[ ] Build the RAG Chain**

  - **Action:** Create the core logic that:
    1.  Accepts a user query.
    2.  Embeds the query using the same sentence-transformer model.
    3.  Performs a similarity search against the vector database to retrieve the top-K relevant chunks.
    4.  Injects the retrieved chunks as context into a prompt for an LLM.
    5.  Calls the LLM API to generate a final answer based on the context.

- **[ ] Test Pipeline in a Notebook**
  - **Action:** Use a Jupyter Notebook to test the end-to-end pipeline with various questions. Debug and ensure all components are working together correctly.

### **Phase 3: Application, Deployment & Documentation**

_This phase makes your project usable and shareable, turning it into a tangible portfolio piece._

- **[ ] Build a User Interface**

  - **Action:** Create a simple web application to interact with your RAG pipeline.
  - **Tech:** `Streamlit` or `Gradio` are highly recommended for their speed and simplicity.

- **[ ] Finalize Dependencies**

  - **Action:** Generate a `requirements.txt` file that lists all necessary packages for your project to run.
  - **Tech:** `pip freeze > requirements.txt`

- **[ ] Deploy the Application**

  - **Action:** Choose a hosting platform and deploy your app.
  - **Action:** Ensure you have configured any necessary API keys as environment variables on the platform.
  - **Tech:** Hugging Face Spaces, Streamlit Community Cloud.

- **[ ] Create Project Documentation**
  - **Action:** Write a comprehensive `README.md` file in your Git repository. It should include:
    - A clear project title and a brief description.
    - The tech stack used.
    - Instructions on how to set up and run the project locally.
    - A live link to your deployed application.

### **Phase 4: Advanced Enhancements (Optional but Recommended)**

_This final phase elevates your project by tackling more complex challenges, demonstrating a deeper level of skill._

- **[ ] Create an Evaluation Set**

  - **Action:** Manually create a list of 15-20 diverse and challenging questions based on the Elden Ring lore, along with the ideal answers. This will be used to objectively measure improvements.

- **[ ] (Advanced) Implement GraphRAG**

  - **Action:** Write a script to identify key entities (e.g., "Ranni," "Caria Manor") and their relationships from your text data.
  - **Action:** Use this to construct and save a knowledge graph.
  - **Action:** Modify your retrieval logic to use the graph to find related context, providing a richer set of information to the LLM.
  - **Tech:** `spaCy` (for Named Entity Recognition), `NetworkX` (for graph building).

- **[ ] Refine and Evaluate**

  - **Action:** Test your advanced GraphRAG system against your evaluation set.
  - **Action:** Experiment with different prompts, embedding models, or retrieval strategies to see if you can improve the accuracy and quality of the answers.

- **[ ] Final Code Review**
  - **Action:** Clean up your code, add comments where necessary, and ensure your project structure is logical and easy to understand. Push all final changes to your Git repository.
