# Customizable RAG Pipeline for Document Collections

This project provides a flexible Retrieval Augmented Generation (RAG) pipeline for querying document collections. It leverages LlamaIndex and sentence-transformer embeddings to build, persist, and query a vector-based index.

## Setup Instructions

**1. Create and Activate Conda Environment:**

```bash
conda env create -f environment.yml
conda activate rag-env # Or your environment name if changed in environment.yml
```

**2. Prepare Your Data Corpus:**

   The system ingests documents from a JSON file, where each object is a document. The path to this file is specified via `corpus_path` in `config.yaml`.
   Ensure your JSON objects contain fields mappable by `src.document_loader.DocumentLoader`, as configured by `corpus_id_field`, `corpus_text_fields`, and `corpus_metadata_fields` in `config.yaml`.

   **Example: Using ACL Anthology BibTeX Data**

   An example workflow is provided for using the ACL Anthology. This involves converting its BibTeX format to the required `data/corpus.json`.

   a.  **Download ACL Anthology Data:**
       ```bash
       mkdir -p data
       wget https://aclanthology.org/anthology+abstracts.bib.gz -P data/
       gunzip data/anthology+abstracts.bib.gz
       # The converter expects data/anthology+abstracts.bib by default
       ```

   b.  **Convert BibTeX to JSON (Example Utility):**
       Compile and run the provided C utility (`bib_to_json.c`):
       ```bash
       gcc -o bib_to_json bib_to_json.c cJSON/cJSON.c -I cJSON -Wall -Wextra -pedantic -std=c99 -lm
       ./bib_to_json data/anthology+abstracts.bib
       ```
       This generates `data/corpus.json`. Adjust the input filename if yours differs.

**3. Configure Settings:**

   a.  **API Credentials (Optional for LLM-based RAG):**
       *   Copy `.env.example` to `.env` (`cp .env.example .env`).
       *   Add your `OPENROUTER_API_KEY` or other LLM provider keys if you intend to integrate LLM generation. The basic retrieval demo functions without this.

   b.  **Main Configuration (`config.yaml`):**
       *   Review and customize `config.yaml`. This file defines corpus paths, storage directories, embedding models, chunking parameters, and other pipeline settings.
       *   Pay special attention to `corpus_path` and `storage_dir`.

## Building the Index

A vector index is required for document retrieval. Choose one of the following methods:

*   **Explicit Build (Recommended):** Use `scripts/build_index.py` for initial creation or to rebuild the index. This script uses settings from `config.yaml`.
    ```bash
    python scripts/build_index.py
    ```
    To force a rebuild (e.g., after data or configuration changes):
    ```bash
    python scripts/build_index.py --rebuild
    ```

*   **Automatic Build:** The chat demo (`scripts/run_chat_demo.py`) will attempt to build the index on its first run if one isn't found in the `storage_dir`. This is convenient but offers less control.

The index is persisted to the `storage_dir` defined in `config.yaml`.

## Running the Chat Demo

With the index built, interact with your documents via the terminal chat demo.

**Prerequisites:**
*   Active Conda environment.
*   Correctly configured `config.yaml`.
*   A built vector index.

**Start the demo:**
```bash
python scripts/run_chat_demo.py
```
This loads the configuration and index, then starts the interactive `Q:` prompt.

**Example Interaction (with ACL Anthology data):**
```
Initializing chat demo...
(Build/load messages...)
--- Chat Demo Started ---
Type 'quit' or 'exit' to end the session.

Q: What are common approaches to machine translation?
Processing query...
A: Based on the retrieved context, common approaches to machine translation include neural machine translation (NMT), which often utilizes encoder-decoder architectures like Transformers, and statistical machine translation (SMT). Attention mechanisms are also highlighted as important in NMT.

  Sources:
    Source 1: (Score: 0.88)
      Title: Attention Is All You Need
      File: N/A
      Preview: Attention Is All You Need Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin Goo...
    Source 2: (Score: 0.85)
      Title: Neural Machine Translation by Jointly Learning to Align and Translate
      File: N/A
      Preview: Neural Machine Translation by Jointly Learning to Align and Translate Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio Université de Montréal Jacobs Univers...

Q: quit
Exiting chat demo.
```
(Responses vary based on your query and indexed corpus.)

## Running Tests

Unit and integration tests are included to verify pipeline components and end-to-end functionality. For detailed information on test structure and execution, see **`tests/README.md`**.

To run all tests quickly from the project root:
```bash
pytest -vv
```

## Core Components Overview

*   **`src/config_loader.py` (`AppConfig`)**: Manages configurations from `config.yaml` and `.env`, providing structured config objects.
*   **`src/document_loader.py` (`DocumentLoader`)**: Loads and transforms documents from the JSON corpus as specified in `config.yaml`.
*   **`src/core_components.py` (`initialize_hf_embedding_model`)**: Initializes the Hugging Face sentence-transformer model (from `config.yaml`) for LlamaIndex.
*   **`src/index_builder.py` (`IndexBuilder`)**: Handles the `VectorStoreIndex` lifecycle: building, loading, and persisting, guided by `config.yaml`.
*   **`src/query_engine_builder.py` (`QueryEngineBuilder`)**: Constructs the LlamaIndex query engine using the built index and query parameters from `config.yaml`.

This project is adaptable for various document collections and retrieval tasks. Consult the source code and docstrings for further details on specific modules.