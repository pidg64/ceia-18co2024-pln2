"""
Dash web application for the Resume RAG Chatbot.
This app provides a web interface to chat with an LLM about resume d        report_progress(90, 'Initializing chatbot...')
        # Initialize chatbot
        chatbot = ResumeRAGChatbot(vector_store, model)
        
        report_progress(100, 'Chatbot initialized successfully!')
        print('Chatbot initialized successfully!')     
        return True, 'Chatbot initialized successfully!'ed in Pinecone using RAG techniques.
"""

import os
import dash
import uuid
import queue
import threading
import dash_bootstrap_components as dbc

from time import sleep
from datetime import datetime
from typing import Tuple, Optional
from pinecone import Pinecone
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dash import dcc, html, Input, Output, State, callback

from resume_management import parse_resume_to_chunks, read_resume_file
from upload_embeddings import (
    load_api_key,
    create_index_if_needed,
    create_vector_store,
    upload_chunks_via_records
)
from llm import ResumeRAGChatbot


# Configuration constants
MODEL_TYPE = os.getenv('MODEL_TYPE', 'openai').lower()
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-5')
MODEL_TEMPERATURE = float(os.getenv('MODEL_TEMPERATURE', '0.1'))
RESUME_FILE_PATH = os.getenv('RESUME_FILE_PATH', 'cvs/resume.txt')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', 768))
INDEX_NAME = os.getenv('INDEX_NAME', 'resume-index')
DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', 625))
DEFAULT_CHUNK_OVERLAP = int(os.getenv('DEFAULT_CHUNK_OVERLAP', 125))
INIT_INTERVAL_MS = 100  # Initialization check interval
RESPONSE_INTERVAL_MS = 500  # Response check interval


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Resume RAG Chatbot'

# Global variables and state
chatbot = None  # ResumeRAGChatbot instance
vector_store = None  # Pinecone vector store instance
pending_responses = {}  # Store for tracking async LLM responses
progress_queue = queue.Queue()  # Queue for progress communication between threads
initialization_thread = None  # Track initialization thread
last_progress_state = {  # Store last known progress to prevent resets
    'progress': 0, 
    'message': 'Initializing chatbot, please wait...'
}


def initialize_chatbot(progress_queue: Optional[queue.Queue] = None) -> Tuple[bool, str]:
    """
    Initialize the chatbot and vector store with progress reporting.
    
    Args:
        progress_queue: Optional queue for progress updates
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    global chatbot, vector_store
    
    def report_progress(percentage: int, message: str):
        """Report progress to queue if available."""
        if progress_queue:
            progress_queue.put({'progress': percentage, 'message': message})
    
    try:
        report_progress(10, 'Starting initialization...')
        
        report_progress(20, 'Loading API keys...')
        model_type = MODEL_TYPE
        if model_type not in ['groq', 'openai']:
            return (
                False,
                f'Unsupported MODEL_TYPE: {model_type}. Use "groq" or "openai".'
            )
        model_name = MODEL_NAME
        model_temperature = MODEL_TEMPERATURE
        
        if model_type == 'groq':
            groq_api_key = load_api_key('GROQ_API_KEY')
            chat_model = ChatGroq
            print(f'Using Groq model: {model_name}')
            model = chat_model(
                model=model_name,
                temperature=model_temperature,
                api_key=SecretStr(groq_api_key)
            )
        elif model_type == 'openai':
            openai_api_key = load_api_key('OPENAI_API_KEY')
            chat_model = ChatOpenAI
            print(f'Using OpenAI model: {model_name}')   
            model = chat_model(
                model=model_name,
                temperature=model_temperature,
                api_key=SecretStr(openai_api_key)
            )        
        report_progress(30, 'Connecting to Pinecone...')
        # Load API key and connect to Pinecone
        pinecone_api_key = load_api_key('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)        
        report_progress(40, 'Creating index...')
        # Get the index
        index = create_index_if_needed(pc, INDEX_NAME)        
        report_progress(50, 'Creating embeddings...')
        # Create embeddings and vector store
        emb = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, 
            dimensions=EMBEDDING_DIMENSIONS
        )
        vector_store = create_vector_store(index, emb)        
        report_progress(60, 'Reading resume file...')
        # Read and parse resume file to get chunks
        resume_file_path = RESUME_FILE_PATH
        print("Reading resume file...")
        resume_text = read_resume_file(resume_file_path)        
        report_progress(70, 'Parsing resume into chunks...')
        print("Parsing resume into chunks...")
        chunks = parse_resume_to_chunks(
            resume_text, 
            chunk_size=DEFAULT_CHUNK_SIZE, 
            overlap=DEFAULT_CHUNK_OVERLAP
        )
        if not chunks:
            print("No chunks were generated. Please check the resume file.")            
        report_progress(80, 'Uploading chunks to vector store...')
        print("Uploading chunks...")
        upload_chunks_via_records(vector_store, chunks)        
        report_progress(90, 'Initializing chatbot...')
        sleep(1)  # Simulate some delay for initialization
        # Initialize chatbot
        chatbot = ResumeRAGChatbot(vector_store, model)        
        report_progress(100, 'Chatbot initialized successfully!')
        sleep(0.1)  # Small delay to ensure progress is captured
        print('Chatbot initialized successfully.')     
        return True, 'Chatbot initialized successfully!'        
    except Exception as e:
        error_msg = f'Error initializing chatbot: {str(e)}'
        report_progress(0, error_msg)
        return False, error_msg


def create_progress_container(progress: int, message: str) -> list:
    """
    Create the initialization container with progress bar.
    
    Args:
        progress: Progress percentage (0-100)
        message: Progress message to display
        
    Returns:
        List of Dash components for the initialization container
    """
    return [
        dbc.Alert(
            [
                html.Div([
                    dbc.Spinner(size='sm', spinner_style={'margin-right': '10px'}),
                    message
                ], style={'display': 'flex', 'align-items': 'center'})
            ],
            color='info',
            style={'margin-bottom': '20px'}
        ),
        dbc.Progress(
            id='init-progress',
            value=progress,
            striped=True,
            animated=True,
            style={'margin-bottom': '20px'}
        )
    ]


def create_processing_message_component(message: str) -> dbc.Card:
    """
    Create a processing message component with spinner.
    
    Args:
        message: The processing message text
        
    Returns:
        A Dash Bootstrap Card component with spinner
    """
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span('🤖', style={'margin-right': '8px'}),
                dbc.Spinner(size='sm', spinner_style={'margin-right': '8px'}),
                html.Span(message)
            ])
        ])
    ], 
    color='light', 
    style={
        'margin': '10px 0',
        'max-width': '80%',
        'margin-left': '0',
        'margin-right': 'auto'
    })


def create_message_component(message: str, is_user: bool) -> dbc.Card:
    """
    Create a message component for the chat interface.
    
    Args:
        message: The message text
        is_user: True if message is from user, False if from bot
        
    Returns:
        A Dash Bootstrap Card component
    """
    if is_user:
        card_color = 'primary'
        icon = '👤'
    else:
        card_color = 'light'
        icon = '🤖'    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'margin-right': '8px'}),
                html.Span(message, style={'white-space': 'pre-wrap'})
            ])
        ])
    ], 
    color=card_color, 
    style={
        'margin': '10px 0',
        'max-width': '80%',
        'margin-left': 'auto' if is_user else '0',
        'margin-right': '0' if is_user else 'auto'
    })


# App layout
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                html.H1(
                    'Resume RAG Chatbot', 
                    className='text-center mb-4',
                    style={'color': '#2c3e50'}
                ),
                html.Hr(),
            ])
        ]),        
        # Initialization progress section
        dbc.Row([
            dbc.Col([
                html.Div(
                    id='initialization-container',
                    children=[
                        dbc.Alert(
                            [
                                html.Div([
                                    dbc.Spinner(size='sm', spinner_style={'margin-right': '10px'}),
                                    'Initializing chatbot, please wait...'
                                ], style={'display': 'flex', 'align-items': 'center'})
                            ],
                            color='info',
                            style={'margin-bottom': '20px'}
                        ),
                        dbc.Progress(
                            id='init-progress',
                            value=0,
                            striped=True,
                            animated=True,
                            style={'margin-bottom': '20px'}
                        )
                    ],
                    style={'display': 'block'}
                )
            ])
        ]),        
        dbc.Row([
            dbc.Col([
                dbc.Alert(
                    id='status-alert',
                    is_open=False,
                    dismissable=True,
                    style={'margin-bottom': '20px'}
                )
            ])
        ]),        
        # Main chat interface (initially hidden)
        dbc.Row([
            dbc.Col([
                html.Div(
                    id='chat-interface',
                    children=[
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4('Chat with the Resume Assistant', className='mb-0')
                            ]),
                            dbc.CardBody([
                                html.Div(
                                    id='chat-container',
                                    style={
                                        'height': 'calc(100vh - 350px)',
                                        'min-height': '200',
                                        'max-height': '600px',
                                        'overflow-y': 'auto',
                                        'border': '1px solid #dee2e6',
                                        'border-radius': '0.375rem',
                                        'padding': '15px',
                                        'background-color': '#f8f9fa',
                                        'resize': 'vertical'
                                    }
                                ),
                                html.Hr(),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id='user-input',
                                        placeholder='Ask about the candidate\'s experience, skills, education...',
                                        type='text',
                                        style={'border-radius': '0.375rem 0 0 0.375rem'}
                                    ),
                                    dbc.Button(
                                        'Send',
                                        id='send-button',
                                        color='primary',
                                        n_clicks=0,
                                        style={'border-radius': '0 0.375rem 0.375rem 0'}
                                    )
                                ], style={'margin-top': '15px'}),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            'Clear Chat',
                                            id='clear-button',
                                            color='secondary',
                                            size='sm',
                                            n_clicks=0,
                                            style={'margin-top': '10px'}
                                        )
                                    ], width=12)
                                ])
                            ])
                        ])
                    ],
                    style={'display': 'none'}  # Initially hidden
                )
            ], width=12)
        ]),        
        # Store for chat history
        dcc.Store(id='chat-history-store', data=[]),
        # Store for initialization state
        dcc.Store(id='init-state-store', data={'initialized': False}),
        # Store for processing state
        dcc.Store(id='processing-store', data={}),
        # Interval for auto-initialization
        dcc.Interval(id='init-interval', interval=INIT_INTERVAL_MS, n_intervals=0, max_intervals=1),
        # Interval for checking LLM responses
        dcc.Interval(id='response-interval', interval=RESPONSE_INTERVAL_MS, n_intervals=0),
    ],
    fluid=True,
    style={'padding': '20px'}
)


# Auto-initialization callback
@callback(
    [
        Output('initialization-container', 'style'),
        Output('chat-interface', 'style'),
        Output('status-alert', 'children'),
        Output('status-alert', 'color'),
        Output('status-alert', 'is_open'),
        Output('init-state-store', 'data')
    ],
    [
        Input('init-interval', 'n_intervals')
    ],
    [
        State('init-state-store', 'data')
    ],
    prevent_initial_call=False
)
def auto_initialize_chatbot(n_intervals, init_state):
    """Automatically initialize the chatbot on app startup using threading."""
    global initialization_thread, progress_queue    
    # Check if already initialized
    if init_state and init_state.get('initialized', False):
        # Already initialized, show chat interface
        return (
            {'display': 'none'},  # Hide initialization container
            {'display': 'block'},  # Show chat interface
            '',  # No alert message
            'info',
            False,  # Alert not open
            init_state
        )    
    # First time initialization
    if n_intervals >= 1 and not initialization_thread:
        try:
            print('Starting threaded initialization...')            
            # Clear any existing progress
            while not progress_queue.empty():
                try:
                    progress_queue.get_nowait()
                except queue.Empty:
                    break            
            # Start initialization in a separate thread
            def threaded_initialization():
                try:
                    success, message = initialize_chatbot(progress_queue)
                    # Put final result in queue
                    progress_queue.put({
                        'progress': 100 if success else 0,
                        'message': message,
                        'completed': True,
                        'success': success
                    })
                except Exception as e:
                    progress_queue.put({
                        'progress': 0,
                        'message': f'Initialization error: {str(e)}',
                        'completed': True,
                        'success': False
                    })            
            initialization_thread = threading.Thread(
                target=threaded_initialization
            )
            initialization_thread.daemon = True
            initialization_thread.start()            
            return (
                {'display': 'block'},  # Show initialization container
                {'display': 'none'},  # Hide chat interface
                '',
                'info',
                False,
                {'initialized': False}
            )            
        except Exception as e:
            print(f'Initialization exception: {str(e)}')
            return (
                {'display': 'block'},
                {'display': 'none'},
                f'Initialization error: {str(e)}',
                'danger',
                True,
                {'initialized': False}
            )    
    # Initial state - show spinner
    if n_intervals == 0:
        print('Initialization pending, showing spinner.')    
    return (
        {'display': 'block'},
        {'display': 'none'},
        '',
        'info',
        False,
        {'initialized': False}
    )


# Progress monitoring callback
@app.callback(
    [
        Output('initialization-container', 'children', allow_duplicate=True),
        Output('initialization-container', 'style', allow_duplicate=True),
        Output('chat-interface', 'style', allow_duplicate=True),
        Output('status-alert', 'children', allow_duplicate=True),
        Output('status-alert', 'color', allow_duplicate=True),
        Output('status-alert', 'is_open', allow_duplicate=True),
        Output('init-state-store', 'data', allow_duplicate=True)
    ],
    [Input('response-interval', 'n_intervals')],
    [
        State('init-state-store', 'data'),
        State('initialization-container', 'style')
    ],
    prevent_initial_call=True
)
def monitor_initialization_progress(n_intervals, init_state, init_container_style):
    """Monitor the progress queue and update UI accordingly."""
    global progress_queue, initialization_thread, last_progress_state
    
    # Skip if already initialized
    if init_state and init_state.get('initialized', False):
        return (
            dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update
        )    
    # Skip if initialization container is hidden
    if init_container_style.get('display') == 'none':
        return (
            dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update
        )    
    # Use last known progress as starting point
    current_progress = last_progress_state['progress']
    current_message = last_progress_state['message']
    completed = False
    success = False
    has_updates = False    
    # Process all available progress updates
    while not progress_queue.empty():
        try:
            progress_data = progress_queue.get_nowait()
            current_progress = progress_data.get('progress', current_progress)
            current_message = progress_data.get('message', current_message)
            completed = progress_data.get('completed', False)
            success = progress_data.get('success', False)
            has_updates = True            
            # Update last known progress state
            last_progress_state['progress'] = current_progress
            last_progress_state['message'] = current_message            
        except queue.Empty:
            break
    
    # If no progress updates found and we have an active thread, use last known state
    if not has_updates and initialization_thread and initialization_thread.is_alive():
        has_updates = True  # Force update to show current state    
    # If no updates at all and no active thread, don't change anything
    if not has_updates:
        return (
            dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update
        )    
    # Update the initialization container with current progress message
    init_container = create_progress_container(current_progress, current_message)
    
    # Handle completion
    if completed:
        initialization_thread = None  # Reset thread tracker
        # Reset progress state for next time
        last_progress_state = {'progress': 0, 'message': 'Initializing chatbot, please wait...'}        
        if success:
            return (
                init_container,
                {'display': 'none'},  # Hide initialization container
                {'display': 'block'},  # Show chat interface
                current_message,  # Success message
                'success',
                True,  # Show success alert
                {'initialized': True}
            )
        else:
            return (
                init_container,
                {'display': 'block'},  # Keep showing initialization
                {'display': 'none'},  # Hide chat interface
                current_message,  # Error message
                'danger',
                True,  # Show error alert
                {'initialized': False}
            )    
    # Return current progress state
    return (
        init_container,
        dash.no_update,  # Keep container visible
        dash.no_update,  # Keep chat hidden
        dash.no_update,  # No alert
        dash.no_update,
        dash.no_update,
        dash.no_update
    )


# Callback for handling chat interactions and clearing chat
@app.callback(
    [
        Output('chat-history-store', 'data'),
        Output('user-input', 'value'),
        Output('processing-store', 'data')
    ],
    [   
        Input('send-button', 'n_clicks'),
        Input('user-input', 'n_submit'),
        Input('clear-button', 'n_clicks')
    ],
    [
        State('user-input', 'value'),
        State('chat-history-store', 'data'),
        State('init-state-store', 'data'),
        State('processing-store', 'data')
    ],
    prevent_initial_call=True
)
def handle_chat_and_clear(
    send_clicks,
    input_submit,
    clear_clicks,
    user_input,
    chat_history,
    init_state,
    processing_state
):
    """Handle user chat input, sending messages, and clearing chat."""
    global pending_responses    
    ctx = dash.callback_context    
    if not ctx.triggered:
        return chat_history or [], user_input or '', processing_state or {}    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]    
    # Handle clear button
    if trigger_id == 'clear-button':
        pending_responses.clear()
        return [], '', {}    
    # Handle send button or enter key
    if trigger_id in ['send-button', 'user-input'] and init_state.get('initialized', False):
        if user_input and user_input.strip() and chatbot is not None:
            chat_history = chat_history or []
            processing_state = processing_state or {}            
            # Add user message immediately
            user_message = {
                'content': user_input.strip(),
                'sender': 'user',
                'timestamp': datetime.now().isoformat()
            }
            chat_history.append(user_message)            
            # Add processing message immediately
            processing_id = str(uuid.uuid4())
            processing_message = {
                'content': 'Processing data...',
                'sender': 'bot',
                'timestamp': datetime.now().isoformat(),
                'processing_id': processing_id,
                'is_processing': True
            }
            chat_history.append(processing_message)            
            # Start async LLM processing
            def process_llm_response():
                try:
                    if chatbot is not None:
                        response = chatbot.chat(user_input.strip())
                        pending_responses[processing_id] = {
                            'content': response,
                            'timestamp': datetime.now().isoformat(),
                            'ready': True
                        }
                    else:
                        pending_responses[processing_id] = {
                            'content': 'Error: Chatbot not initialized',
                            'timestamp': datetime.now().isoformat(),
                            'ready': True
                        }
                except Exception as e:
                    pending_responses[processing_id] = {
                        'content': f'Error: {str(e)}',
                        'timestamp': datetime.now().isoformat(),
                        'ready': True
                    }            
            # Start the thread
            thread = threading.Thread(target=process_llm_response)
            thread.daemon = True
            thread.start()            
            # Update processing state
            processing_state[processing_id] = {
                'started': True,
                'user_input': user_input.strip()
            }            
            return chat_history, '', processing_state    
    return chat_history or [], user_input or '', processing_state or {}


# Callback for updating LLM responses
@app.callback(
    Output('chat-history-store', 'data', allow_duplicate=True),
    [Input('response-interval', 'n_intervals')],
    [
        State('chat-history-store', 'data'),
        State('processing-store', 'data')
    ],    
    prevent_initial_call=True
)
def update_llm_responses(n_intervals, chat_history, processing_state):
    """Check for completed LLM responses and update chat history."""
    global pending_responses    
    if not chat_history or not pending_responses:
        return chat_history or []    
    updated_history = chat_history.copy()
    updated = False    
    # Check each message for processing completion
    for i, message in enumerate(updated_history):
        if (message.get('is_processing', False) and 
            message.get('processing_id') in pending_responses):            
            processing_id = message['processing_id']
            response_data = pending_responses[processing_id]            
            if response_data.get('ready', False):
                # Replace processing message with actual response
                updated_history[i] = {
                    'content': response_data['content'],
                    'sender': 'bot',
                    'timestamp': response_data['timestamp']
                }
                # Clean up
                del pending_responses[processing_id]
                updated = True    
    return updated_history if updated else chat_history


@app.callback(
    Output('chat-container', 'children'),
    Input('chat-history-store', 'data')
)
def update_chat_display(chat_history):
    """Update the chat display with conversation history."""    
    if not chat_history:
        return html.Div([
            html.P(
                'Welcome! Start asking questions about the candidate\'s resume.',
                className='text-muted text-center',
                style={'margin-top': '50px'}
            )
        ])    
    chat_messages = []
    for message in chat_history:        
        if message['sender'] == 'user':
            chat_messages.append(
                create_message_component(message['content'], is_user=True)
            )
        else:
            # Check if this is a processing message
            if message.get('is_processing', False):
                # Create a processing message with spinner
                chat_messages.append(
                    create_processing_message_component(message['content'])
                )
            else:
                # Regular bot message
                chat_messages.append(
                    create_message_component(message['content'], is_user=False)
                )    
    return html.Div(
        chat_messages,
        style={'height': '100%'}
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)