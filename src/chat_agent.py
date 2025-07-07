from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import Config

def get_chat_agent(vector_db):
    """Create a chat agent using OpenAI GPT-4 and the vector DB as retriever."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Create the LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create the prompt template
    template = """You are a helpful AI assistant that answers questions based on the provided context from a knowledge graph.

Context information:
{context}

Question: {question}

Please answer the question based on the context provided. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate in your response.

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain using LCEL
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def chat_with_agent(agent, user_input):
    """Get response from the agent for user input."""
    return agent.invoke(user_input) 