import numpy as np, faiss, pickle, json, bibtexparser
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .config import DATA_DIR, INDEX_PATH
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
from .utils import text_from_pdf, clean

def _papers():
    for bib in (DATA_DIR/"bibtex").rglob("*.bib"):
        entries = [e for e in bibtexparser.load(open(bib)).entries]
        for idx, entry in enumerate(entries, 0):  
            pdf = DATA_DIR/"pdf"/bib.parent.name/(f"{idx}.pdf")
            if pdf.exists():
                yield str(idx), entry.get("title", ""), entry.get("abstract", ""), pdf

def build():
    model = SentenceTransformer(EMBED_MODEL)
    vecs, meta = [], []
    for pid, title, abs_, pdf in _papers():
        print(f"Processing {pid}...")
        vecs.append(model.encode(f"{title} {abs_}", normalize_embeddings=True))
        meta.append({"id":pid,"title":title,"abstract":abs_})
    vecs = np.stack(vecs)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure vectorspace dir exists
    faiss.write_index(faiss.IndexFlatIP(vecs.shape[1]).add(vecs), str(INDEX_PATH))
    pickle.dump(meta, open(str(INDEX_PATH)+".meta","wb"))
    print(f"Indexed {len(meta)} papers → {INDEX_PATH}") 