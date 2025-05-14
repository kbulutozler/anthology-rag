import numpy as np, faiss, pickle
from sentence_transformers import SentenceTransformer
from .config import INDEX_PATH
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(str(INDEX_PATH)+".meta","rb") as f:
            self.meta  = pickle.load(f)
        self.st    = SentenceTransformer(EMBED_MODEL)
    def search(self, q, k=3):
        qv = self.st.encode(q, normalize_embeddings=True)
        D,I = self.index.search(np.expand_dims(qv,0),k)
        return [self.meta[i]|{"score":float(s)} for i,s in zip(I[0],D[0])] 