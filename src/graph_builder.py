import networkx as nx
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from src.config import Config
from typing import Any
import streamlit as st

class SimpleKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = []
        self.edges = []
    
    def add_node(self, node_id, label=None):
        if node_id not in self.nodes:
            self.nodes.append(node_id)
            self.graph.add_node(node_id, label=label or node_id)
    
    def add_edge(self, source, target, relation=None):
        edge = (source, target, relation) if relation else (source, target)
        if edge not in self.edges:
            self.edges.append(edge)
            self.graph.add_edge(source, target, relation=relation)

def build_knowledge_graph(content_type: str, content: Any) -> SimpleKnowledgeGraph:
    """Build a knowledge graph using LangChain's latest Knowledge Graph RAG pipeline."""
    if content_type == 'txt':
        text_content = content
    elif content_type == 'json':
        text_content = str(content)
    else:
        raise ValueError('Unsupported content type for graph building.')
    
    # Use LangChain's latest LLMGraphTransformer for entity/relation extraction
    if not Config.OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found in environment variables")
        return SimpleKnowledgeGraph()
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create transformer with specific node and relationship types for better extraction
    transformer = LLMGraphTransformer(
        llm=llm
    )
    
    try:
        # Convert text to Document format
        documents = [Document(page_content=text_content)]
        
        # Extract graph data
        graph_documents = transformer.convert_to_graph_documents(documents)
        
        if not graph_documents or len(graph_documents) == 0:
            st.warning("No graph data was extracted. Try a different or larger file.")
            return SimpleKnowledgeGraph()
        
        # Convert to our SimpleKnowledgeGraph for compatibility
        kg = SimpleKnowledgeGraph()
        
        # Extract nodes from the graph documents
        for node in graph_documents[0].nodes:
            kg.add_node(node.id, node.id)
        
        # Extract relationships from the graph documents
        for relationship in graph_documents[0].relationships:
            kg.add_edge(relationship.source.id, relationship.target.id, relationship.type)
        
        if len(kg.nodes) == 0:
            st.warning("No entities or relationships were extracted by LangChain Knowledge Graph RAG. Try a different or larger file.")
        
        return kg
        
    except Exception as e:
        st.error(f"LangChain Knowledge Graph RAG error: {e}")
        return SimpleKnowledgeGraph() 