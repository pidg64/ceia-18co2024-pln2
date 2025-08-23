"""
Resume Q&A Agent CLI

A command-line interface for querying candidate resumes using LangGraph,
OpenAI, and Pinecone vector database. The agent extracts candidate names
from user questions, retrieves relevant resume information, and provides
conversational responses.
"""

import os

from pinecone import Pinecone
from dotenv import load_dotenv
from langgraph.types import Send
from langchain_core.documents import Document
from pydantic import SecretStr, BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables.config import RunnableConfig
from typing import Annotated, List, Optional, TypedDict, Sequence
from langchain_pinecone import PineconeVectorStore, PineconeRerank
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

# Load environment variables
load_dotenv()

# Configuration constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', 768))
MESSAGE_HISTORY_LIMIT = int(os.getenv('MESSAGE_HISTORY_LIMIT', 4))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')


def initialize_clients() -> tuple[ChatOpenAI, OpenAIEmbeddings, Pinecone]:
    """
    Initialize and return the LLM, embeddings, and Pinecone clients.
    
    Returns:
        tuple: A tuple containing (llm, embeddings, pinecone_client)
        
    Raises:
        ValueError: If required API keys are not found in environment variables
    """
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError(
            'Please set your OPENAI_API_KEY and PINECONE_API_KEY ' \
            'environment variables.'
        )
        
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1,
        api_key=SecretStr(OPENAI_API_KEY)
    )
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        api_key=SecretStr(OPENAI_API_KEY)
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return llm, embeddings, pc


# Initialize clients
llm, embeddings, pc = initialize_clients()


class Candidates(BaseModel):
    """
    Pydantic model for structured output of candidate name extraction.
    
    Attributes:
        names (List[str]): List of candidate names mentioned in user's query
    """
    names: List[str] = Field(
        description='A list of candidate names mentioned in the user\'s query.'
    )


planner = llm.with_structured_output(Candidates)


def append_or_reset(prev: list[str], upd: Optional[list[str]]) -> list[str]:
    """
    State reducer function that either resets or appends to context list.
    
    Args:
        prev (list[str]): Previous context list
        upd (Optional[list[str]]): Update to apply (None means reset)
        
    Returns:
        list[str]: Empty list if upd is None, otherwise concatenated lists
    """
    return [] if upd is None else prev + upd


class State(TypedDict):
    """
    Main state schema for the LangGraph workflow.
    
    Attributes:
        candidates (List[str]): List of candidate names extracted from user query
        context (Annotated[List[str], append_or_reset]): Retrieved resume contexts
        messages (Annotated[Sequence[BaseMessage], add_messages]): Conversation history
    """
    candidates: List[str]
    context: Annotated[List[str], append_or_reset]
    messages: Annotated[Sequence[BaseMessage], add_messages]


class WorkerState(TypedDict):
    """
    State schema for individual retrieval workers.
    
    Attributes:
        question (str): The user's question to search for
        candidate (str): Name of the candidate to retrieve information for
    """
    question: str
    candidate: str


def clear_state(state: State) -> dict:
    """
    Clears the context from the previous conversation turn.
    
    This function resets the context to prevent information leakage
    between different user questions in the same conversation session.
    
    Args:
        state (State): Current state of the conversation
        
    Returns:
        dict: Update with context reset to empty list
    """
    print('>>> Clearing previous state.')
    return {'context': None}


def orchestrator(state: State) -> dict:
    """
    Extracts candidate names from the user's question using LLM.
    
    This function analyzes the latest message in the conversation history
    to identify any candidate names mentioned in the user's query.
    
    Args:
        state (State): Current state containing conversation messages
        
    Returns:
        dict: Update containing list of extracted candidate names
    """
    question = state['messages'][-1].content
    print('>>> Extracting candidates from the question.')
    candidate_model = planner.invoke([
        SystemMessage(
            content=(
                'You are an expert at extracting candidate names from a '
                'user query.'
            )
        ),
        HumanMessage(
            content=(
                'Extract the full names of any candidates mentioned in the '
                f'following query: "{question}"'
            )
        )
    ])
    candidates = getattr(candidate_model, 'names', [])
    return {'candidates': candidates}


def _create_index_name(candidate: str) -> str:
    """
    Generate Pinecone index name from candidate name.
    
    Args:
        candidate (str): Full name of the candidate
        
    Returns:
        str: Sanitized index name in format 'firstname-lastname-index'
    """
    return f'{candidate.lower().replace(" ", "-")}-index'


def _retrieve_and_rerank_documents(
        vector_store: PineconeVectorStore, 
        question: str, 
        candidate: str
    ) -> Sequence[Document]:
    """
    Retrieve and rerank documents for a candidate based on the question.
    
    Args:
        vector_store (PineconeVectorStore): Pinecone vector store instance
        question (str): User's question for context retrieval
        candidate (str): Name of the candidate
        
    Returns:
        Sequence[Document]: Sequence of reranked documents
    """
    reranker = PineconeRerank(
        model='bge-reranker-v2-m3',
        top_n=3
    )
    retriever = vector_store.as_retriever(
        search_kwargs={'k': 5, 'score_threshold': 0.35}
    )
    retrieved_docs = retriever.invoke(question)
    print(f'>>> Retrieved {len(retrieved_docs)} documents for {candidate}.')
    print('>>> Reranking documents.')
    reranked_docs = reranker.compress_documents(
        documents=retrieved_docs,
        query=question
    )
    print(f'>>> Reranked {len(reranked_docs)} documents for {candidate}.')
    return reranked_docs


def retrieval_worker(state: WorkerState) -> dict:
    """
    Retrieves and reranks context for a specific candidate.
    
    This function handles the retrieval of relevant resume information
    for a given candidate. It supports a special 'NoCandidate' case
    for general queries that don't mention specific candidates.
    
    Args:
        state (WorkerState): State containing candidate name and question
        
    Returns:
        dict: Update containing formatted context for the candidate
    """
    candidate = state['candidate']
    if candidate == 'NoCandidate':
        return {
            'context': [
                'No candidate specified in the question. '
                'Just answer based on your general knowledge.'
            ]
        }
    
    question = state['question']
    index_name = _create_index_name(candidate)
    print(
        f'>>> Retrieving context for candidate {candidate} '
        f'using index {index_name}.'
    )
    
    try:
        if index_name not in pc.list_indexes().names():
            return {'context': [f'No resume found for {candidate}.']}
            
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )
        
        reranked_docs = _retrieve_and_rerank_documents(
            vector_store, question, candidate
        )
        
        context = "\n\n---\n\n".join(
            [doc.page_content for doc in reranked_docs]
        )
        formatted_context = f"Context for {candidate}:\n{context}"
        return {'context': [formatted_context]}
        
    except Exception as e:
        print(f'>>> Error retrieving context for {candidate}: {e}.')
        return {
            'context': [
                f'An error occurred while retrieving resume information '
                f'for {candidate}.'
            ]
        }

def _format_conversation_history(messages: Sequence[BaseMessage]) -> str:
    """
    Format recent conversation history for inclusion in system prompt.
    
    Args:
        messages (Sequence[BaseMessage]): List of conversation messages
        
    Returns:
        str: Formatted conversation history string
    """
    last_messages_raw = (
        messages[-MESSAGE_HISTORY_LIMIT-1:] 
        if len(messages) >= MESSAGE_HISTORY_LIMIT + 1
        else messages[:-1]
    )
    last_messages = ''
    for msg in last_messages_raw:
        persona = 'Human' if isinstance(msg, HumanMessage) else 'AI'
        last_messages += f"{persona}: {msg.content}\n"
    return last_messages


def _create_system_prompt(contexts: str, conversation_history: str) -> str:
    """
    Create the system prompt for the responder LLM.
    
    Args:
        contexts (str): Concatenated resume contexts
        conversation_history (str): Formatted conversation history
        
    Returns:
        str: Complete system prompt with instructions and context
    """
    return (
        'You are a helpful assistant for a resume Q&A system. Your goal is '
        'to answer the user\'s question based on the provided context from '
        'one or more resumes.\n\n'
        '**Instructions for your response:**\n'
        '1. **Conversational Tone:** Answer in a natural, fluid, and '
        'conversational manner.\n'
        '2. **Synthesize Information:** If context from multiple resumes is '
        'provided, synthesize the information into a single, coherent '
        'response. Do not just list the findings for each person '
        'separately.\n'
        '3. **Avoid Lists:** Do not use bullet points or lists. Formulate '
        'the answer as a complete sentence or paragraph.\n'
        '4. **Handle Missing Information:** If the context does not contain '
        'the answer, state that the information is not available in the '
        'resume(s) in a polite way.\n\n'
        'Answer the user\'s question accurately.'
        f'\n\nContexts:\n\n{contexts}'
        f'\n\nLast {MESSAGE_HISTORY_LIMIT} messages:\n\n{conversation_history}'
    )


def responder(state: State) -> dict:
    """
    Generates the final answer based on retrieved contexts and conversation history.
    
    This function synthesizes information from multiple resume contexts
    and previous conversation turns to provide a comprehensive,
    conversational response to the user's question.
    
    Args:
        state (State): Current state containing messages and contexts
        
    Returns:
        dict: Update containing the AI's response message
    """
    question = state['messages'][-1].content
    contexts = "\n\n".join(state['context'])
    
    conversation_history = _format_conversation_history(state['messages'])
    system_message = _create_system_prompt(contexts, conversation_history)
    human_message = f"Question: {question}"
    
    print('>>> Generating answer based on retrieved contexts.')
    response = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ])
    return {'messages': [response]}

def assign_workers(state: State) -> List[Send]:
    """
    Assigns parallel retrieval workers to each identified candidate.
    
    This function creates a Send instruction for each candidate found
    in the user's query, enabling parallel document retrieval. If no
    candidates are found, it assigns a single 'NoCandidate' worker.
    
    Args:
        state (State): Current state containing list of candidates
        
    Returns:
        List[Send]: List of Send instructions for parallel worker execution
    """
    candidates = state['candidates'] or ['NoCandidate']
    print(f'>>> Assigning work to workers for candidates: {candidates}.')
    
    return [
        Send(
            'retrieval_worker',
            {
                'candidate': candidate,
                'question': state['messages'][-1].content
            }
        )
        for candidate in candidates
    ]


def _build_graph() -> StateGraph:
    """
    Build and configure the LangGraph state graph.
    
    Returns:
        StateGraph: Configured graph ready for compilation
    """
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node('clear_state', clear_state)
    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_node("retrieval_worker", retrieval_worker)
    graph_builder.add_node("responder", responder)

    # Define edges
    graph_builder.add_edge(START, 'clear_state')
    graph_builder.add_edge('clear_state', 'orchestrator')
    graph_builder.add_conditional_edges(
        "orchestrator", 
        assign_workers, 
        {"retrieval_worker": "retrieval_worker"}
    )
    graph_builder.add_edge("retrieval_worker", "responder")
    graph_builder.add_edge("responder", END)
    
    return graph_builder


def main():
    """
    Main function to build the graph and run the CLI chatbot.
    
    This function initializes the conversation agent with memory,
    compiles the graph, and runs an interactive CLI loop for
    user queries about resume information.
    """
    # Build and compile the graph with memory
    graph_builder = _build_graph()
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
