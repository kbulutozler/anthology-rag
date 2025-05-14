#!/usr/bin/env bash
set -e

# Parse command-line flags
DOWNLOAD_PDFS=false
DO_INDEXING=false
for arg in "$@"; do
  case "$arg" in
    --download-pdfs) DOWNLOAD_PDFS=true ;;
    --do-indexing) DO_INDEXING=true ;;
    *) ;;
  esac
done

# Create or update Conda environment
if conda info --envs | grep -q 'llm-agent'; then
  echo "Conda env 'llm-agent' already exists. Skipping update."
else
  echo "Creating Conda env 'llm-agent'..."
  conda env create -f environment.yml
fi

echo "✅ Env ready. Activate with: conda activate llm-agent"

# Conditional PDF download
if [ "$DOWNLOAD_PDFS" = true ]; then
  echo "📑 Fetching NAACL-2025 PDFs & BibTeX..."
  conda run -n llm-agent python -m scripts.download_naacl2025
else
  echo "⏭️  Skipping PDF download (assumed already downloaded)."
fi

# Conditional FAISS indexing
if [ "$DO_INDEXING" = true ]; then
  echo "🔨 Building FAISS index..."
  conda run -n llm-agent python -m scripts.build_index
else
  echo "⏭️  Skipping indexing (assumed already done)."
fi

echo "activate the environment with: conda activate llm-agent"

echo "to start the agent, run: uvicorn scripts.run_agent:app --host 0.0.0.0 --port 9001"

echo "after the agent is running, to start the chat, in another terminal run: python3 scripts/chat.py"

