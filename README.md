# NAACL-2025 Research Assistant

This repository provides a terminal-based research assistant for NAACL-2025 papers. It downloads papers, builds a FAISS index over embeddings, and serves an LLM via FastAPI that can answer questions by retrieving relevant paper titles.

## Setup Instructions

Follow these steps to get the research assistant up and running:

**1. Clone the Repository:**

   First, clone this repository to your local machine:
   ```bash
   git clone <repo-url> # Replace <repo-url> with the actual URL of this repository
   cd llm-agent       # Or your repository's root folder name
   ```

**2. Configure Your Settings:**

   Setting up your project involves two simple configuration parts:

   a.  **Provide Your API Credentials (Secret):**
       *   After cloning, you should find a template file named `.env.example` in the project root. (If it's missing, please create it with the content shown in the troubleshooting section below).
       *   Create your personal secrets file by copying this template:
           ```bash
           cp .env.example .env
           ```
       *   Now, open the new `.env` file (e.g., with a text editor like VS Code, Nano, or Notepad).
       *   Inside `.env`, replace the placeholder value for `OPENROUTER_API_KEY` with your actual OpenRouter API key.
           For example, it should look something like this after editing:
           ```env
           # .env (This is your local file with your secrets)
           OPENROUTER_API_KEY="sk-or-v1-your-actual-key-goes-here"
           ```
       *   **Important:** Your `.env` file contains your private secrets and is already listed in `.gitignore`, so it will not (and should not) be committed to Git.

   b.  **Configure Your AI Model and Base URL (Non-Secret):**
       *   The `config.yaml` file (also in the project root) specifies the AI model the application will use (e.g., `model_name: "google/gemini-flash-1.5"`) and the `OPENROUTER_BASE_URL` (e.g., `OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1"`).
       *   This file is included in the repository with default values. If you wish to use a different model from OpenRouter or need to change the base URL, you can directly edit these values in `config.yaml`.

   Once you've created and filled your `.env` file and (optionally) adjusted `config.yaml`, you're ready for the next step.

**3. Bootstrap Environment & Process Data:**

   Bootstrap the Conda environment and optionally download PDFs and build the search index:
   ```bash
   chmod +x setup.sh
   ./setup.sh [--download-pdfs] [--do-indexing]
   ```
   *   `--download-pdfs`: Download NAACL-2025 PDFs & BibTeX files.
   *   `--do-indexing`: Build the FAISS search index from the downloaded papers.
   *   If you run `./setup.sh` without flags, it assumes the data is already downloaded and indexed.

## Running the Agent

1. Activate the environment:
   ```bash
   conda activate llm-agent
   ```
2. Start the FastAPI backend:
   ```bash
   uvicorn scripts.run_agent:app --port 9001
   ```

3. Chat with the agent using the provided Python CLI script:
   ```bash
   python3 scripts/chat.py
   ```

## Troubleshooting

*   **`.env.example` file is missing?**
    If the `.env.example` file is not present in the project root after cloning, you can create it by running the following command in your terminal (ensure you are in the project's root directory):
    ```bash
    echo -e "# .env.example\n# This is an example file. Copy it to .env and fill in your actual API key.\n# The .env file is gitignored and will not be committed.\n# OPENROUTER_BASE_URL is now configured in config.yaml\n\nOPENROUTER_API_KEY=\"YOUR_OPENROUTER_API_KEY_HERE\"" > .env.example
    ```
    After running this command, the `.env.example` file will be created. You can then proceed to copy it to `.env` as described in the setup instructions (i.e., `cp .env.example .env`).

## File Structure

- `config.yaml` – application settings, like the AI model name and OpenRouter base URL (version-controlled).
- `.env.example` – a template showing the required environment variables for secrets (e.g., API key) (version-controlled).
- `.env` – your personal file for API keys and other secrets (created locally from `.env.example`, gitignored).
- `data/` – raw NAACL-2025 PDFs & BibTeX
- `vectorspace/`