import umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load embedding model and FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)

# Extract all vectors from FAISS index
vectors = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
docs = list(vectorstore.docstore._dict.values())

# Reduce embeddings to 2D using UMAP
reducer = umap.UMAP()
embedding_2d = reducer.fit_transform(vectors)

# Try to label by source file (from metadata)
sources = [doc.metadata.get('source', 'unknown') for doc in docs]
unique_sources = list(set(sources))
source_colors = {src: cm.rainbow(i / len(unique_sources)) for i, src in enumerate(unique_sources)}
colors = [source_colors[src] for src in sources]

# Try to enable interactive tooltips
try:
    import mplcursors
    interactive = True
except ImportError:
    print("🔧 Optional: Run 'pip install mplcursors' for tooltips.")
    interactive = False

# Plot with labels
plt.figure(figsize=(14, 9))
sc = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, alpha=0.7, edgecolors='k')

# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=src,
                      markerfacecolor=source_colors[src], markersize=8) for src in unique_sources]
plt.legend(handles=handles, title="Source Files", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("UMAP Projection of Document Embeddings (Colored by Source)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.tight_layout()

# Add hover tooltips if mplcursors is installed
if interactive:
    cursor = mplcursors.cursor(sc, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(docs[sel.index].page_content[:250] + "..."))

plt.show()
