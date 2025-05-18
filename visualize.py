import umap
import numpy as np
import matplotlib.pyplot as plt
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model and FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.load_local("vectorstore", embedding_model)
vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
# Extract all vectors from the FAISS index
vectors = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)

# Optional: Extract corresponding document texts
docs = [doc.page_content for doc in vectorstore.docstore._dict.values()]

# Use UMAP to reduce dimensionality to 2D
reducer = umap.UMAP()
embedding_2d = reducer.fit_transform(vectors)

# Plot the 2D embeddings
plt.figure(figsize=(12, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6, edgecolors='k')
plt.title("UMAP Projection of Document Embeddings")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()
