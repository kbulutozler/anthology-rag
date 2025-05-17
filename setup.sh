#!/bin/bash

# Anthology RAG Project Setup Script
# This script guides you through the initial setup of the project.

echo "Starting Anthology RAG Project Setup..."

# Function to check if a command exists
command_exists () {
    type "$1" &> /dev/null ;
}

# 1. Check for Conda
if ! command_exists conda; then
    echo "Error: Conda is not installed or not in your PATH."
    echo "Please install Miniconda or Anaconda from https://docs.conda.io/en/latest/miniconda.html and try again."
    exit 1
fi
echo "[1/6] Conda check: OK"

# 2. Create and Activate Conda Environment
ENV_NAME="anthology-rag"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists."
else
    echo "Creating Conda environment '${ENV_NAME}' from environment.yml..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create Conda environment. Please check environment.yml and try again."
        exit 1
    fi
    echo "Conda environment '${ENV_NAME}' created successfully."
fi

echo "To activate the environment, run: conda activate ${ENV_NAME}"
echo "(You might need to run this manually in your current shell after the script finishes)"
echo "Attempting to activate for subsequent steps (may not affect parent shell)..."

# Try to activate for the script's context, though it won't persist for the user's shell
# This is mainly for subsequent compilation steps if any were dependent on env paths right away.
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "Warning: Failed to activate Conda environment in script. Some subsequent steps might fail if they depend on the env."
    echo "Please ensure you activate it manually ('conda activate ${ENV_NAME}') before running other scripts."
fi
echo "[2/6] Conda environment setup: Done (activate manually if needed)"


# 3. Prepare ACL Anthology Data
read -p "Do you want to download the ACL Anthology BibTeX data now? (requires wget, gunzip) [y/N]: " download_data
if [[ "$download_data" =~ ^[Yy]$ ]]; then
    echo "Downloading ACL Anthology data..."
    mkdir -p data
    wget https://aclanthology.org/anthology+abstracts.bib.gz -P data/
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download anthology+abstracts.bib.gz. Please check your internet connection or download manually."
    else
        echo "Download complete. Extracting..."
        gunzip data/anthology+abstracts.bib.gz
        if [ $? -ne 0 ]; then
            echo "Error: Failed to extract data/anthology+abstracts.bib.gz."
        else
            echo "Extraction complete: data/anthology+abstracts.bib"
        fi
    fi
fi

BIB_FILE="data/anthology+abstracts.bib"
CORPUS_JSON_FILE="data/corpus.json"

if [ ! -f "$BIB_FILE" ]; then
    echo "Warning: ACL Anthology BibTeX file ($BIB_FILE) not found."
    echo "Please download it manually (e.g., from https://aclanthology.org/anthology+abstracts.bib.gz), extract, and place it as $BIB_FILE."
fi

read -p "Do you want to compile bib_to_json and convert $BIB_FILE to $CORPUS_JSON_FILE now? (requires gcc) [y/N]: " compile_c
if [[ "$compile_c" =~ ^[Yy]$ ]]; then
    if ! command_exists gcc; then
        echo "Error: GCC (C compiler) is not installed or not in your PATH."
        echo "Please install gcc (e.g., 'sudo apt-get install build-essential' on Debian/Ubuntu) and try again."
    elif [ ! -f "$BIB_FILE" ]; then
        echo "Error: Cannot compile and run bib_to_json because $BIB_FILE is missing."
    else
        echo "Compiling bib_to_json..."
        gcc -o bib_to_json bib_to_json.c cJSON/cJSON.c -I cJSON -Wall -Wextra -pedantic -std=c99 -lm
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compile bib_to_json.c. Please check for errors."
        else
            echo "Compilation successful. Converting $BIB_FILE to $CORPUS_JSON_FILE..."
            ./bib_to_json "$BIB_FILE"
            if [ $? -ne 0 ]; then
                echo "Error: Failed to convert BibTeX to JSON. The bib_to_json program might have encountered an issue."
            elif [ -f "$CORPUS_JSON_FILE" ]; then
                echo "Conversion successful: $CORPUS_JSON_FILE has been created."
            else
                 echo "Error: Conversion seemed to run but $CORPUS_JSON_FILE was not created."
            fi
        fi
    fi
fi
echo "[3/6] Data preparation: Done (manual steps may be required if skipped or failed)"

# 4. Configure Environment Variables
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo ".env file created. Please edit it to add your OPENROUTER_API_KEY if you plan to use OpenRouter LLMs."
    echo "The basic retrieval demo will work without an API key."
else
    echo ".env file already exists. Please ensure it is correctly configured."
fi
echo "[4/6] Environment variables (.env): Done"

# 5. Review Main Configuration
echo "Please review the main configuration file: config.yaml."
echo "Ensure 'corpus_path' points to your JSON corpus (e.g., data/corpus.json)."
echo "Ensure 'storage_dir' points to your desired index storage location (default: ./storage)."
echo "[5/6] Main configuration (config.yaml): Review recommended"

# 6. Build Initial Index (Optional)
read -p "Do you want to build the initial vector index now? (This can take a while) [y/N]: " build_index_now
if [[ "$build_index_now" =~ ^[Yy]$ ]]; then
    echo "Building the index using 'python scripts/build_index.py --rebuild' ..."
    echo "Please ensure your Conda environment '${ENV_NAME}' is active in this terminal, or activate it and run the script manually."
    # Attempt to run if conda activate worked in script context
    python scripts/build_index.py --rebuild
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build index. Make sure '${ENV_NAME}' is active and all paths in config.yaml are correct."
    else
        echo "Index build process initiated. Check terminal output for status."
    fi
else
    echo "Skipping initial index build. You can build it later with 'python scripts/build_index.py'"
    echo "Alternatively, 'scripts/run_chat_demo.py' will attempt to build it on first run if not found."
fi
echo "[6/6] Initial index build: Done (or skipped)"

echo "
Setup complete! Next steps:
1. Ensure Conda environment is active: conda activate ${ENV_NAME}
2. If not done, verify/complete data preparation (download .bib, convert to .json).
3. Verify/edit .env with your API keys (if using LLMs).
4. Verify/edit config.yaml (especially corpus_path, storage_dir).
5. If index was not built, run: python scripts/build_index.py
6. Run the chat demo: python scripts/run_chat_demo.py
7. Run tests: pytest -vv
" 