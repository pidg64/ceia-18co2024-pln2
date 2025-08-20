import os
import operator

from pinecone import Pinecone
from dotenv import load_dotenv
from langgraph.types import Send
from typing import Annotated, List, TypedDict
from pydantic import SecretStr, BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', 768))

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

class State(TypedDict):
    question: str
    candidates: List[str]
    context: Annotated[List[str], operator.add]
    answer: str

class WorkerState(TypedDict):
    question: str
    candidate: str

def orchestrator(state: State) -> dict:
    question = state['question']
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
    question = state['question']
    index_name = f'{candidate.lower().replace(" ", "-")}-index'
    print(
        f'>>> Retrieving context for candidate {candidate} '
        f'using index {index_name}'
        )
    try:
        if index_name not in pc.list_indexes().names():
            return {'context': [f'No resume found for {candidate}.']}
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )
        retriever = vector_store.as_retriever(
            search_kwargs={'k': 3, 'score_threshold': 0.35}
        )
        retrieved_docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )
        formatted_context = f"Context for {candidate}:\n{context}"
        return {'context': [formatted_context]}
    except Exception:
        return {
            'context': [
                'An error occurred while retrieving resume information '
                'for {candidate}.'
            ]
        }

def responder(state: State) -> dict:
    question = state['question']
    contexts = "\n\n".join(state['context'])
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
    )
    human_message = f"Question: {question}\n\nContexts:\n{contexts}"
    print('>>> Generating answer based on retrieved contexts.')
    response = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ])
    return {'answer': response.content}

def assign_workers(state: State) -> List[Send]:
    """
    Assigns a worker to each identified candidate.
    """
    print(
        f'>>> Assigning work to workers for candidates: {state["candidates"]}'
    )
    # The `Send` tool allows us to trigger multiple parallel node executions
    return [
        Send(
            'retrieval_worker',
            {
                'candidate': candidate,
                'question': state['question']
            }
        )
        for candidate in state['candidates']
    ]


def main():
    """
    Main function to build the graph and run the CLI chatbot.
    """
    # Build the graph
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_node("retrieval_worker", retrieval_worker)
    graph_builder.add_node("responder", responder)

    # Define edges
    graph_builder.add_edge(START, "orchestrator")
    graph_builder.add_conditional_edges(
        "orchestrator", 
        assign_workers, 
        {"retrieval_worker": "retrieval_worker"}
    )
    graph_builder.add_edge("retrieval_worker", "responder")
    graph_builder.add_edge("responder", END)

    # Compile the graph
    resume_agent = graph_builder.compile()

    print('Agent: How can I help you today?')
    while True:
        user_input = input('User: ').strip()
        if not user_input:
            continue
        if user_input.lower() in {'exit', 'quit'}:
            print('Agent: Goodbye!')
            break
        
        # Invoke the agent with the user's question
        result = resume_agent.invoke({'question': user_input})
        print(f'Agent: {result["answer"]}')

if __name__ == '__main__':
    main()
