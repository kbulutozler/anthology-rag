# ACL Anthology Research Assistant

This repository provides a terminal-based research assistant for the ACL Anthology. 

## Setup Instructions

**1. Clone the Repository:**

   ```bash
   git clone https://github.com/kbulutozler/anthology-rag.git
   cd anthology-rag
   ```

**2. Process the ACL Anthology Bib File:**

   Download the full ACL Anthology BibTeX+abstracts file and extract it:
   ```bash
   wget https://aclanthology.org/anthology+abstracts.bib.gz
   gunzip anthology+abstracts.bib.gz
   mv anthology+abstracts.bib data/
   ```
   Convert the BibTeX file to JSON for faster processing:
   ```bash
   gcc -o bib_to_json bib_to_json.c cJSON/cJSON.c -I cJSON -Wall -Wextra -pedantic -std=c99
   ./bib_to_json data/anthology+abstracts.bib
   ```
   End result is json file containing all the available bibtex+abstracts for the entire ACL Anthology.

**3. Configure Your Settings:**

   a.  **Provide Your API Credentials (Secret):**
       *   Copy the template `.env.example` to `.env` and fill in your OpenRouter API key as before.

   b.  **Configure Your AI Model and Base URL (Non-Secret):**
       *   Edit `config.yaml` if you wish to change the model or API endpoint.

**4. Create and Activate Conda Environment:**

   Create the Conda environment from the `environment.yml` file and activate it:
   ```bash
   conda env create -f environment.yml
   conda activate anthology-rag
   ```

## How Indexing Works

The indexing process converts the ACL Anthology data (from `data/anthology+abstracts.json`) into a searchable format that the RAG system can efficiently query. This is handled by the `scripts/build_index.py` script and involves the following key steps:

1.  **Embedding Model Initialization**: 
    *   A local sentence-transformer model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) is initialized using `src.core_components.initialize_embedding_model`. This model converts text (titles, abstracts) into numerical vectors (embeddings) that capture semantic meaning.
    *   This embedding model is set globally for LlamaIndex via `Settings.embed_model`.

2.  **Data Loading**: 
    *   The `src.data_loader.AnthologyLoader` reads the `anthology+abstracts.json` file.
    *   Each paper entry is transformed into a LlamaIndex `Document` object. The main text of the `Document` typically combines the paper's title and abstract, while other details (authors, year, URL) are stored as metadata.

3.  **Index Construction**: 
    *   A `VectorStoreIndex` is created from the loaded `Document` objects. 
    *   During this process, LlamaIndex internally: 
        *   Chunks the text of each `Document` into smaller `Node` objects.
        *   Uses the configured embedding model to generate vector embeddings for each `Node`.
        *   Stores these nodes and their embeddings in a vector store (by default, a simple in-memory store that gets persisted).

4.  **Persistence**: 
    *   The constructed `VectorStoreIndex` (including the vector store, document store, and index metadata) is saved to disk in the `./storage/` directory.
    *   This allows the index to be quickly loaded in subsequent runs without needing to reprocess all the data, unless a rebuild is explicitly forced (e.g., using `python scripts/build_index.py --rebuild`).

To build or update the index, run:
```bash
python scripts/build_index.py
```
To force a rebuild of the index (e.g., if the source data or embedding model changes):
```bash
python scripts/build_index.py --rebuild
```

## How Querying Works (Retrieval Logic)

Querying the ACL Anthology data involves retrieving relevant documents from the persisted index based on a user's query. This process is primarily managed by functions within `src/core_components.py`:

1.  **Initialize Embedding Model**:
    *   As with indexing, the same sentence-transformer embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) must be active. The `initialize_embedding_model()` function ensures this and sets it globally in `llama_index.core.Settings.embed_model`.

2.  **Load the Index**:
    *   The `load_anthology_index(storage_dir)` function is used to load the previously built and persisted `VectorStoreIndex` from the specified storage directory (defaulting to `./storage/`).
    *   This function first ensures the embedding model is initialized (as the index structure and its vectors are tied to the model used at creation) and then loads the index components (docstore, vector store, etc.) via LlamaIndex's `StorageContext` and `load_index_from_storage`.
    *   If the index is not found at the specified location, it returns `None` and logs a message.

3.  **Create a Retriever**:
    *   Once the `VectorStoreIndex` is loaded, the `get_anthology_retriever(index, similarity_top_k)` function is called.
    *   This function creates a retriever object from the index (typically using `index.as_retriever()`).
    *   The `similarity_top_k` parameter (defaulting to 3) determines how many of the most similar documents (nodes) should be fetched from the index for a given query.

4.  **Retrieve Relevant Nodes**:
    *   The obtained retriever object has a `retrieve("your query string")` method.
    *   When called with a user's query string, the retriever:
        *   First, converts the query string into an embedding vector using the globally set embedding model.
        *   Then, queries the vector store (part of the loaded index) to find the nodes whose embeddings are most similar to the query's embedding.
        *   It returns a list of `NodeWithScore` objects, each containing a retrieved `Node` (a chunk of a document) and its similarity score to the query.

These retrieved nodes (containing text excerpts and metadata from the original papers) form the context that will be passed to a Large Language Model (LLM) in the subsequent RAG (Retrieval-Augmented Generation) stage to generate an answer or summary.

## How the Full RAG Process Works (Query Logic & Generation)

When you use the chat interface (`scripts/chat.py`), the system performs the following steps to answer your query:

1.  **Initialization (Once at Startup)**:
    *   **Configuration Loading**: Reads `config.yaml` for model settings and `.env` for your OpenRouter API key.
    *   **Embedding Model**: Initializes the sentence-transformer model (e.g., `all-MiniLM-L6-v2`) specified in `src/core_components.py` and sets it globally via `Settings.embed_model`. This is the same model used for indexing.
    *   **LLM Initialization**: Initializes the Large Language Model client (e.g., `LlamaOpenAI` configured for your chosen OpenRouter model like `google/gemini-flash-1.5`) using your API key and sets it globally via `Settings.llm`.
    *   **Index Loading**: Loads the persisted `VectorStoreIndex` from the `./storage/` directory using `load_anthology_index`.
    *   **Retriever Creation**: Creates a retriever from the loaded index using `get_anthology_retriever`, configured to fetch a certain number of top similar document chunks (e.g., `similarity_top_k=3`).
    *   **Query Engine Creation**: Combines the index, retriever, and the initialized LLM into a `QueryEngine` using `get_query_engine`. This engine orchestrates the RAG process.

2.  **Per User Query (In the Chat Loop)**:
    *   **User Input**: You type your question into the terminal.
    *   **Retrieval**: 
        *   The `QueryEngine` passes your query text to its internal `Retriever`.
        *   The `Retriever` converts your query to an embedding vector (using the global embedding model) and searches the `VectorStoreIndex` for the document chunks (nodes) with the most similar embeddings.
        *   A list of relevant `NodeWithScore` objects (containing text and metadata) is returned.
    *   **Context Augmentation & Prompting**:
        *   The `QueryEngine` takes these retrieved nodes (the context) and your original query.
        *   It then formulates a prompt that typically includes:
            *   Instructions to the LLM (e.g., "Answer the question based on the following context...").
            *   The retrieved context (text from the relevant paper chunks).
            *   Your original query.
    *   **LLM Interaction**:
        *   The `QueryEngine` sends this augmented prompt to the configured LLM (e.g., Gemini Flash via OpenRouter).
    *   **Response Generation**:
        *   The LLM processes the prompt and generates a response based *both* on its general knowledge and, more importantly, on the specific context provided from the ACL Anthology papers.
        *   **Output to User**:
            *   The `scripts/chat.py` script prints the LLM's generated answer to your terminal.
            *   It also lists the source nodes (title, authors, year, URL, and retrieval score) that were used as context, allowing you to see where the information came from.

This cycle of retrieval, context augmentation, prompting, and generation allows the system to provide answers grounded in the content of the ACL Anthology.

## Chatting with the Anthology Assistant

Once you have successfully built the index (see "How Indexing Works"), you can start a chat session with the RAG system.

**Prerequisites:**
*   Ensure your Conda environment (`anthology-rag`) is activated: `conda activate anthology-rag`
*   Ensure you have a valid OpenRouter API key in your `.env` file.
*   Ensure you have run `python scripts/build_index.py` at least once to create the searchable index in the `./storage/` directory.

**To start chatting, run:**

```bash
python scripts/chat.py
```

The script will guide you through the initialization steps (loading config, models, index). Once ready, you can type your questions about ACL Anthology papers. 

Type `quit` or `exit` to end the chat session.

**Example Interaction:**

```
You: What are some common approaches to machine translation?

Thinking...

Assistant:
Based on the provided context, common approaches to machine translation include neural machine translation (NMT), which often utilizes encoder-decoder architectures like Transformers, and statistical machine translation (SMT), which relies on statistical models learned from bilingual text corpora. Some papers also discuss hybrid approaches and the importance of attention mechanisms in NMT.

--- Sources ---
Source 1 (Score: 0.8765):
  Title: Attention Is All You Need
  Authors: Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones, Llion, Gomez, Aidan N, Kaiser, Lukasz, Polosukhin, Illia
  Year: 2017
  URL: https://aclanthology.org/P17-1001/
-------------------------------
Source 2 (Score: 0.8521):
  Title: Neural Machine Translation by Jointly Learning to Align and Translate
  Authors: Bahdanau, Dzmitry, Cho, Kyunghyun, Bengio, Yoshua
  Year: 2015
  URL: https://aclanthology.org/P15-1001/
-------------------------------
# ... (other sources)
--------------------------------------------------------------------------------
You: 
```

(Note: The exact LLM response and sources will vary based on your query, the content of your index, and the LLM model used.)

## Running Tests

To ensure the components of the RAG system are functioning correctly, unit tests are provided in the `tests/`