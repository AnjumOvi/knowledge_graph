import streamlit as st
from src.config import Config
from src.file_handler import parse_uploaded_file
from src.graph_builder import build_knowledge_graph
from src.graph_visualizer import visualize_graph
from src.vector_store import store_graph_in_vector_db
from src.chat_agent import get_chat_agent, chat_with_agent

st.set_page_config(page_title="Graph RAG App", layout="wide")
st.title("🕸️ Knowledge Graph RAG with GPT-4 & ChromaDB")

# File upload
uploaded_file = st.file_uploader("Upload a .txt or .json file", type=["txt", "json"])

if uploaded_file:
    content_type, content = parse_uploaded_file(uploaded_file)
    with st.spinner("Generating knowledge graph (this may take a minute for large files)..."):
        kg = build_knowledge_graph(content_type, content)
    st.subheader("Knowledge Graph Visualization")
    visualize_graph(kg)
    st.success("Knowledge graph constructed and visualized.")
    if len(kg.nodes) > 0:
        with st.spinner("Storing graph in ChromaDB and preparing chat agent..."):
            db = store_graph_in_vector_db(kg)
            st.success("Graph stored in ChromaDB.")
            agent = get_chat_agent(db)
        st.subheader("Chat with your Knowledge Graph")
        user_input = st.text_input("Ask a question:")
        if user_input:
            with st.spinner("Thinking..."):
                response = chat_with_agent(agent, user_input)
            st.markdown(f"**Agent:** {response}")
    else:
        st.warning("No knowledge graph was generated from your file. Please try a different or larger file.") 