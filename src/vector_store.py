from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from src.config import Config
import os

def store_graph_in_vector_db(kg):
    """Store knowledge graph nodes in ChromaDB with OpenAI embeddings."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    embeddings = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY, model="text-embedding-3-large")
    texts = [kg.graph.nodes[node]['label'] for node in kg.nodes]
    metadatas = [{"type": "graph_node", "node_id": node} for node in kg.nodes]
    db = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=Config.CHROMA_DB_PATH)
    db.persist()
    return db 