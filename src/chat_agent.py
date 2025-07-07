from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from src.config import Config

def get_chat_agent(vector_db):
    """Create a chat agent using OpenAI GPT-4 and the vector DB as retriever."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    llm = OpenAI(api_key=Config.OPENAI_API_KEY, model_name="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())
    return qa_chain

def chat_with_agent(agent, user_input):
    """Get response from the agent for user input."""
    return agent.run(user_input) 