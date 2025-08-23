"""
Resume Q&A Agent GUI

A web interface for querying candidate resumes using LangGraph,
OpenAI, and Pinecone vector database. The agent extracts candidate names
from user questions, retrieves relevant resume information, and provides
conversational responses through a Dash web application.
"""

import dash
import uuid
import threading
import dash_bootstrap_components as dbc

from typing import Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dash import dcc, html, Input, Output, State, callback
from langchain_core.runnables.config import RunnableConfig

from langgraph_flow import build_graph

# Load environment variables
load_dotenv()

# Configuration constants
RESPONSE_INTERVAL_MS = 500  # Response check interval in milliseconds

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Resume Q&A Agent'

# Global variables and state
resume_agent = None  # LangGraph resume agent instance
pending_responses = {}  # Store for tracking async LLM responses


def initialize_agent() -> Tuple[bool, str]:
    """
    Initialize the LangGraph resume agent.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    global resume_agent
    
    try:
        print('Initializing LangGraph resume agent...')
        
        # Build and compile the graph with memory
        graph_builder = build_graph()
        memory = MemorySaver()
        resume_agent = graph_builder.compile(checkpointer=memory)
        
        print('Resume agent initialized successfully!')
        return True, 'Resume agent initialized successfully!'
        
    except Exception as e:
        error_msg = f'Error initializing resume agent: {str(e)}'
        print(error_msg)
        return False, error_msg


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
                    'Resume Q&A Agent', 
                    className='text-center mb-4',
                    style={'color': '#2c3e50'}
                ),
                html.P(
                    'Ask questions about candidate resumes. You can mention specific candidate names in your questions.',
                    className='text-center mb-4 text-muted'
                ),
                html.Hr(),
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
        
        # Main chat interface
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4('Chat with the Resume Assistant', className='mb-0')
                    ]),
                    dbc.CardBody([
                        html.Div(
                            id='chat-container',
                            style={
                                'height': 'calc(100vh - 350px)',
                                'min-height': '300px',
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
                                placeholder='Ask about candidates: "What is John\'s experience?" or "Compare John and Mary\'s skills"',
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
            ], width=12)
        ]),        
        
        # Store for chat history
        dcc.Store(id='chat-history-store', data=[]),
        # Store for agent configuration (thread_id)
        dcc.Store(id='agent-config-store', data={'thread_id': str(uuid.uuid4())}),
        # Store for processing state
        dcc.Store(id='processing-store', data={}),
        # Interval for checking LLM responses
        dcc.Interval(id='response-interval', interval=RESPONSE_INTERVAL_MS, n_intervals=0),
    ],
    fluid=True,
    style={'padding': '20px'}
)


# Initialize agent on startup
@callback(
    [
        Output('status-alert', 'children'),
        Output('status-alert', 'color'),
        Output('status-alert', 'is_open')
    ],
    [Input('response-interval', 'n_intervals')],
    prevent_initial_call=False
)
def initialize_on_startup(n_intervals):
    """Initialize the agent on app startup."""
    global resume_agent
    
    if resume_agent is None and n_intervals == 0:
        success, message = initialize_agent()
        if success:
            return message, 'success', True
        else:
            return message, 'danger', True
    
    return dash.no_update, dash.no_update, dash.no_update


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
        State('agent-config-store', 'data'),
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
    agent_config,
    processing_state
):
    """Handle user chat input, sending messages, and clearing chat."""
    global pending_responses, resume_agent    
    
    ctx = dash.callback_context    
    
    if not ctx.triggered:
        return chat_history or [], user_input or '', processing_state or {}    
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]    
    
    # Handle clear button
    if trigger_id == 'clear-button':
        pending_responses.clear()
        # Reset agent with new thread_id
        agent_config['thread_id'] = str(uuid.uuid4())
        return [], '', {}    
    
    # Handle send button or enter key
    if trigger_id in ['send-button', 'user-input'] and resume_agent is not None:
        if user_input and user_input.strip():
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
                'content': 'Processing your question...',
                'sender': 'bot',
                'timestamp': datetime.now().isoformat(),
                'processing_id': processing_id,
                'is_processing': True
            }
            chat_history.append(processing_message)            
            
            # Start async LLM processing
            def process_agent_response():
                try:
                    if resume_agent is not None:
                        # Define config for conversation persistence
                        config = RunnableConfig({
                            'configurable': {'thread_id': agent_config['thread_id']}
                        })
                        
                        # Invoke the agent with the user's question
                        result = resume_agent.invoke(
                            {
                                'messages': [HumanMessage(user_input.strip())],
                                'candidates': [],
                                'context': [],
                            },
                            config
                        )
                        
                        response_content = result["messages"][-1].content
                        pending_responses[processing_id] = {
                            'content': response_content,
                            'timestamp': datetime.now().isoformat(),
                            'ready': True
                        }
                    else:
                        pending_responses[processing_id] = {
                            'content': 'Error: Resume agent not initialized',
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
            thread = threading.Thread(target=process_agent_response)
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
def update_agent_responses(n_intervals, chat_history, processing_state):
    """Check for completed agent responses and update chat history."""
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
                'Welcome! Ask questions about candidate resumes. You can mention specific names like "What is John\'s experience?" or ask comparative questions.',
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
    app.run(debug=True, host='0.0.0.0', port=8051)
