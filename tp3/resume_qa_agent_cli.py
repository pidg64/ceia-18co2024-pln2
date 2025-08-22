import os
import operator

from pinecone import Pinecone
from dotenv import load_dotenv
from langgraph.types import Send
from pydantic import SecretStr, BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, Optional, TypedDict, Sequence
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore, PineconeRerank
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', 768))
MESSAGE_HISTORY_LIMIT = int(os.getenv('MESSAGE_HISTORY_LIMIT', 4))

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

class Candidates(BaseModel):
    names: List[str] = Field(
        description='A list of candidate names mentioned in the user\'s query.'
    )

planner = llm.with_structured_output(Candidates)

def append_or_reset(prev: list[str], upd: Optional[list[str]]) -> list[str]:
    # Reset when None; otherwise append
    return [] if upd is None else prev + upd

class State(TypedDict):
    question: str
    candidates: List[str]
    context: Annotated[List[str], append_or_reset]
    messages: Annotated[Sequence[BaseMessage], add_messages]

class WorkerState(TypedDict):
    question: str
    candidate: str

def clear_state(state: State) -> dict:
    """Clears the context and candidates from the previous turn."""
    print('>>> Clearing previous state.')
    return {'context': None, 'candidates': []}

def orchestrator(state: State) -> dict:
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
    candidates = candidate_model.names
    return {'candidates': candidates}

def retrieval_worker(state: WorkerState) -> dict:
    candidate = state['candidate']
    if candidate == 'NoCandidate':
        return {
            'context': [
                'No candidate specified in the question. '
                'Just answer based on your general knowledge.'
            ]
        }
    question = state['question']
    index_name = f'{candidate.lower().replace(" ", "-")}-index'
    print(
        f'>>> Retrieving context for candidate {candidate} '
        f'using index {index_name}.'
        )
    try:
        if index_name not in pc.list_indexes().names():
            return {'context': [f'No resume found for {candidate}.']}
        reranker = PineconeRerank(
            model='bge-reranker-v2-m3',
            top_n=3
        )
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
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
        print(
            f'>>> Reranked {len(reranked_docs)} documents for {candidate}.'
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
                'An error occurred while retrieving resume information '
                'for {candidate}.'
            ]
        }

def responder(state: State) -> dict:
    question = state['messages'][-1].content
    contexts = "\n\n".join(state['context'])
    last_messages_raw = (
        state['messages'][-MESSAGE_HISTORY_LIMIT-1:] 
        if len(state['messages']) >= MESSAGE_HISTORY_LIMIT + 1
        else state['messages'][:-1]
    )
    last_messages = ''
    for msg in last_messages_raw:
        persona = 'Human' if isinstance(msg, HumanMessage) else 'AI'
        last_messages += f"{persona}: {msg.content}\n"
    system_message = (
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
        f'\n\nLast {MESSAGE_HISTORY_LIMIT} messages:\n\n{last_messages}'
    )
    human_message = f"Question: {question}"
    print('>>> Generating answer based on retrieved contexts.')
    response = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ])
    return {'messages': [response]}

def assign_workers(state: State) -> List[Send]:
    """
    Assigns a worker to each identified candidate.
    """
    candidates = state['candidates'] or ['NoCandidate']
    print(
        f'>>> Assigning work to workers for candidates: {candidates}.'
    )
    # The `Send` tool allows us to trigger multiple parallel node executions
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


def main():
    """
    Main function to build the graph and run the CLI chatbot.
    """
    # Build the graph
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

    # Add memory saver
    memory = MemorySaver()
    # Compile the graph
    resume_agent = graph_builder.compile(checkpointer=memory)

    # Define default config
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
        result = resume_agent.invoke({'messages': [HumanMessage(user_input)]}, config)
        print(f'>>> History size: {len(result["messages"])}.')
        print(f'Agent: {result["messages"][-1].content}')

if __name__ == '__main__':
    main()
