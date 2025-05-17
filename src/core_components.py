import torch
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional

# Default embedding model name
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def initialize_hf_embedding_model(model_name: Optional[str] = None) -> HuggingFaceEmbedding:
    """
    Initialize and set the global embedding model for LlamaIndex.
    """
    if model_name is None:
        model_name = DEFAULT_EMBED_MODEL_NAME

    # Determine the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check against global Settings
    if Settings.embed_model is not None and getattr(Settings.embed_model, 'model_name', None) == model_name and getattr(Settings.embed_model, 'device', None) == device:
        print(f"Embedding model {model_name} is already initialized and set on {device}.")
        return Settings.embed_model

    print(f"Initializing embedding model: {model_name} on {device}")
    embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
    Settings.embed_model = embed_model
    print(f"Successfully set {model_name} as the global LlamaIndex embedding model (using {device}).")
    return embed_model 