"""
Resume Q&A Agent CLI

A command-line interface for querying candidate resumes using LangGraph,
OpenAI, and Pinecone vector database. The agent extracts candidate names
from user questions, retrieves relevant resume information, and provides
conversational responses.
"""

import os

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage

from langgraph_flow import build_graph

# Load environment variables
load_dotenv()

# Configuration constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', 768))
MESSAGE_HISTORY_LIMIT = int(os.getenv('MESSAGE_HISTORY_LIMIT', 4))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')


def main():
    """
    Main function to build the graph and run the CLI chatbot.
    
    This function initializes the conversation agent with memory,
    compiles the graph, and runs an interactive CLI loop for
    user queries about resume information.
    """
    # Build and compile the graph with memory
    graph_builder = build_graph()
    memory = MemorySaver()
    resume_agent = graph_builder.compile(checkpointer=memory)
    # Define default config for conversation persistence
    config = RunnableConfig({'configurable': {'thread_id': 'default'}})
    print('Agent: How can I help you today?')
    while True:
        user_input = input('User: ').strip()
        if not user_input:
            continue
        if user_input.lower() in {'exit', 'quit'}:
            print('Agent: Goodbye!')
            break        
        # Invoke the agent with the user's question
        result = resume_agent.invoke(
            {
                'messages': [HumanMessage(user_input)],
                'candidates': [],
                'context': [],
            },
            config
        )
        print(f'>>> History size: {len(result["messages"])}.')
        print(f'Agent: {result["messages"][-1].content}')


if __name__ == '__main__':
    main()
