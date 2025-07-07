import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

def visualize_graph(kg) -> None:
    """Visualize the knowledge graph using networkx and matplotlib."""
    G = kg.graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=2000,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            arrows=True,
            arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    st.image(f"data:image/png;base64,{img_str}", use_container_width=True)
    st.write(f"**Graph Statistics:**")
    st.write(f"- Number of nodes: {len(G.nodes)}")
    st.write(f"- Number of edges: {len(G.edges)}")
    plt.close() 

