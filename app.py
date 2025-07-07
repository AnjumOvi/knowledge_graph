import streamlit as st
from src.config import Config
from src.file_handler import parse_uploaded_file
from src.graph_builder import build_knowledge_graph
from src.graph_visualizer import visualize_graph
from src.vector_store import store_graph_in_vector_db
from src.chat_agent import get_chat_agent, chat_with_agent

st.set_page_config(page_title="Graph RAG App", layout="wide")
st.title("🕸️ Knowledge Graph RAG with GPT-4 & ChromaDB")

# Initialize session state
if 'chat_agent' not in st.session_state:
    st.session_state.chat_agent = None
if 'current_file_hash' not in st.session_state:
    st.session_state.current_file_hash = None
if 'graph_built' not in st.session_state:
    st.session_state.graph_built = False

# File upload
uploaded_file = st.file_uploader("Upload a .txt or .json file", type=["txt", "json"])

if uploaded_file:
    # Check if this is a new file
    file_hash = hash(uploaded_file.getvalue())
    
    if file_hash != st.session_state.current_file_hash:
        # New file uploaded - rebuild everything
        st.session_state.current_file_hash = file_hash
        st.session_state.graph_built = False
        st.session_state.chat_agent = None
        
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
                st.session_state.chat_agent = get_chat_agent(db)
                st.session_state.graph_built = True
        else:
            st.warning("No knowledge graph was generated from your file. Please try a different or larger file.")
    
    # Show chat interface if graph was built
    if st.session_state.graph_built and st.session_state.chat_agent:
        st.subheader("Chat with your Knowledge Graph")
        user_input = st.text_input("Ask a question:")
        if user_input:
            with st.spinner("Thinking..."):
                response = chat_with_agent(st.session_state.chat_agent, user_input)
            st.markdown(f"**Agent:** {response}")
    elif st.session_state.current_file_hash and not st.session_state.graph_built:
        st.warning("No knowledge graph was generated from your file. Please try a different or larger file.")

# Show instructions if no file is uploaded
else:
    st.info("👆 Please upload a .txt or .json file to get started!")
    st.markdown("""
    ### How it works:
    1. **Upload** a text or JSON file
    2. **Wait** for the knowledge graph to be generated
    3. **Chat** with your knowledge graph using natural language
    4. **Ask questions** about the content in your file
    
    The app uses LangChain's latest Knowledge Graph RAG pipeline to extract entities and relationships from your text, then stores them in ChromaDB for fast retrieval and chat.
    """) 