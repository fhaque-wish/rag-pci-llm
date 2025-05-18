from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Ollama
import config

def run_qa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(config.VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

    llm = Ollama(model="gemma:2b")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    print("\n🔍 Ask questions about your documents (type 'exit' to quit)\n")
    while True:
        query = input("❓ Question: ")
        if query.lower() == "exit":
            break
        result = qa(query)
        print(f"\n💡 Answer: {result['result']}\n")

if __name__ == "__main__":
    run_qa()
