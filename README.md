# 🕸️ Knowledge Graph RAG with GPT-4 & ChromaDB

A one-of-a-kind Streamlit app to build, visualize, and chat with a knowledge graph RAG from your `.txt` or `.json` files, powered by LangChain, OpenAI GPT-4, and ChromaDB.

## Features
- Upload `.txt` or `.json` files
- Automatic knowledge graph construction (LangChain Knowledge Graph RAG)
- Interactive graph visualization
- Store graph in ChromaDB with OpenAI embeddings
- Chat interface powered by OpenAI GPT-4, grounded in your knowledge graph

## Setup
1. **Clone the repo**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up your `.env` file**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   CHROMA_DB_PATH=chroma_db
   ```
4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Project Structure
```
app.py
requirements.txt
.env
src/
  ├── config.py
  ├── file_handler.py
  ├── graph_builder.py
  ├── graph_visualizer.py
  ├── vector_store.py
  ├── chat_agent.py
  └── utils.py
```

## Notes
- Only `.txt` and `.json` files are supported.
- All API keys and sensitive info should be stored in `.env`.
- Built with best practices for modularity, security, and scalability. 