# 🔐 PCI DSS RAG Assistant with LangChain, FAISS & Ollama

This project is a **Retrieval-Augmented Generation (RAG)** system designed to help organizations interactively query PCI DSS documents (PDF, DOCX, TXT) using a local LLM served by Ollama (e.g., `gemma:2b`). It demonstrates a full ingestion-to-response pipeline with document chunking, embedding, vector search, and LLM-based generation.

---

## 🧠 Features

- 🔍 **Semantic search** over PCI DSS and related documents
- 📄 Ingests `.pdf`, `.txt`, `.docx` documents
- 🧱 Chunking + embedding using `MiniLM` (384D vectors)
- 💾 Fast similarity search via FAISS
- 🤖 LLM response generation via `Ollama` + LangChain
- 📊 UMAP visualization of embeddings
- ⚙️ Clean CLI interaction and modular codebase

---

## 🖼️ Architecture

```
+-------------------------------+
| Source Documents (.pdf, etc) |
+-------------------------------+
              |
       [ Ingestion ]
              |
+------------------------------+
| Chunking + Embedding (384D) |
+------------------------------+
              |
        FAISS Vector DB
              ▲
              |
     [ User Enters Query ]
              |
+------------------------------+
|  Retrieve Top-k Chunks from |
|         FAISS DB            |
+------------------------------+
              |
+------------------------------+
| LLM (Ollama: gemma:2b)       |
|   Generates RAG response     |
+------------------------------+
              |
       [ Display in CLI ]
```

---

## 📁 Folder Structure

```
rag-lightweight-qa/
├── data/                  # Input documents (.pdf, .docx, .txt)
├── vectorstore/           # Saved FAISS index (generated)
├── main.py                # CLI chatbot interface
├── ingest.py              # Ingest and embed documents
├── visualize.py           # 2D UMAP visualization
├── config.py              # Chunking config
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/rag-pci-llm.git
cd rag-pci-llm
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama & Pull a Model

```bash
ollama run gemma:2b  # or mistral, llama3, etc.
```

### 5. Ingest your documents

Place your `.pdf`, `.txt`, or `.docx` files in the `data/` folder, then run:

```bash
python ingest.py
```

### 6. Start querying

```bash
python main.py
```

### 7. Visualize Embeddings (optional)

```bash
python visualize.py
```

---

## 📦 Dependencies

- Python 3.11+
- LangChain
- FAISS (CPU)
- HuggingFace sentence-transformers
- Ollama (LLM server)
- UMAP-learn (for visualization)

---

## 📜 License

MIT License

---

## ✨ Credits

- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com)
- PCI Security Standards Council

---

## 📌 TODOs

- [ ] Add Streamlit web interface
- [ ] Filter by PCI version / control ID
- [ ] Export response history
- [ ] Integrate query reranking

---

## 🙋‍♂️ Questions or Contributions?

Feel free to open an issue or PR!