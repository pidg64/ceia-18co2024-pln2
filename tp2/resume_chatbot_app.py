"""
Dash web application for the Resume RAG Chatbot.
This app provides a web interface to chat with an LLM about resume data
stored in Pinecone using RAG techniques.
"""

import os
import dash

from datetime import datetime
from pinecone import Pinecone
from pydantic import SecretStr
from langchain_groq import ChatGroq
import dash_bootstrap_components as dbc
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dash import dcc, html, Input, Output, State, callback

from upload_embeddings import (
    load_api_key,
    create_index_if_needed,
    create_vector_store
)
from llm import ResumeRAGChatbot


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Resume RAG Chatbot'

# Global variables
chatbot = None
vector_store = None


def initialize_chatbot():
    """Initialize the chatbot and vector store."""
    global chatbot, vector_store
    
    try:
        model_type = os.getenv('MODEL_TYPE', 'openai').lower()
        if model_type not in ['groq', 'openai']:
            return (
                False,
                f'Unsupported MODEL_TYPE: {model_type}. Use "groq" or "openai".'
            )
        model_name = os.getenv('MODEL_NAME', 'gpt-5')
        model_temperature = float(os.getenv('MODEL_TEMPERATURE', '0.1'))
        if model_type == 'groq':
            api_key = load_api_key('GROQ_API_KEY')
            chat_model = ChatGroq
            print(f'Using Groq model: {model_name}')
        elif model_type == 'openai':
            openai_api_key = load_api_key('OPENAI_API_KEY')
            chat_model = ChatOpenAI
            print(f'Using OpenAI model: {model_name}')   
        model = chat_model(
            model=model_name,
            temperature=model_temperature,
            api_key=SecretStr(openai_api_key) if model_type == 'openai' else None
        )
        # Load API key and connect to Pinecone
        pinecone_api_key = load_api_key('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)        
        # Get the index
        index_name = 'resume-index'
        index = create_index_if_needed(pc, index_name)        
        # Create embeddings and vector store
        emb = OpenAIEmbeddings(
            model='text-embedding-3-small', 
            dimensions=768
        )
        vector_store = create_vector_store(index, emb)        
        # Initialize chatbot
        chatbot = ResumeRAGChatbot(vector_store, model)   
        print('Chatbot initialized successfully.')     
        return True, 'Chatbot initialized successfully!'
        
    except Exception as e:
        return False, f'Error initializing chatbot: {str(e)}'


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
        text_color = 'white'
        alignment = 'end'
        icon = '👤'
    else:
        card_color = 'light'
        text_color = 'dark'
        alignment = 'start'
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
app.layout = dbc.Container([
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
                                    'height': '400px',
                                    'overflow-y': 'auto',
                                    'border': '1px solid #dee2e6',
                                    'border-radius': '0.375rem',
                                    'padding': '15px',
                                    'background-color': '#f8f9fa'
                                }
                            ),
                            html.Hr(),
                            dbc.InputGroup([
                                dbc.Input(
                                    id='message-input',
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
    # Interval for auto-initialization
    dcc.Interval(id='init-interval', interval=100, n_intervals=0, max_intervals=1),

], fluid=True, style={'padding': '20px'})


# Auto-initialization callback
@callback(
    [
        Output('initialization-container', 'style'),
        Output('chat-interface', 'style'),
        Output('init-progress', 'value'),
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
    """Automatically initialize the chatbot on app startup."""
    # Check if already initialized
    if init_state and init_state.get('initialized', False):
        # Already initialized, show chat interface
        return (
            {'display': 'none'},  # Hide initialization container
            {'display': 'block'},  # Show chat interface
            100,  # Progress at 100%
            '',  # No alert message
            'info',
            False,  # Alert not open
            init_state
        )
    
    # First time initialization
    if n_intervals >= 1:
        try:
            print('Starting initialization...')
            success, message = initialize_chatbot()
            if success:
                print('Hiding initialization container and showing chat interface.')
                return (
                    {'display': 'none'},  # Hide initialization container
                    {'display': 'block'},  # Show chat interface
                    100,  # Progress complete
                    message,  # Success message
                    'success',
                    True,  # Show success alert
                    {'initialized': True}
                )
            else:
                print('Initialization failed, showing error message.')
                return (
                    {'display': 'block'},  # Keep showing initialization
                    {'display': 'none'},  # Hide chat interface
                    0,  # Reset progress
                    message,  # Error message
                    'danger',
                    True,  # Show error alert
                    {'initialized': False}
                )
        except Exception as e:
            print(f'Initialization exception: {str(e)}')
            return (
                {'display': 'block'},
                {'display': 'none'},
                0,
                f'Initialization error: {str(e)}',
                'danger',
                True,
                {'initialized': False}
            )
    
    # Initial state - show spinner
    print('Initialization in progress, showing spinner.')
    return (
        {'display': 'block'},
        {'display': 'none'},
        50,  # Partial progress
        '',
        'info',
        False,
        {'initialized': False}
    )


@callback(
    [
        Output('chat-history-store', 'data'),
        Output('message-input', 'value')
    ],
    [
        Input('send-button', 'n_clicks'),
        Input('message-input', 'n_submit'),
        Input('clear-button', 'n_clicks')
    ],
    [
        State('message-input', 'value'),
        State('chat-history-store', 'data')
    ],
    prevent_initial_call=True
)
def handle_chat_interactions(
    send_clicks,
    message_submit,
    clear_clicks,
    message,
    chat_history
    ):
    """Handle chat interactions including sending messages and clearing."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return chat_history, message or ''
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Clear chat history
    if trigger_id == 'clear-button' and clear_clicks > 0:
        if chatbot:
            chatbot.clear_history()
        return [], ''
    
    # Send message
    if trigger_id in ['send-button', 'message-input'] and message and message.strip():
        if chatbot is None:
            # Add error message to chat
            error_entry = {
                'user_message': message,
                'bot_response': 'Chatbot is still initializing. Please wait a moment and try again.',
                'timestamp': datetime.now().isoformat()
            }
            chat_history.append(error_entry)
            return chat_history, ''
        
        try:
            print(f'User prompt: {message.strip()}')
            # Get bot response
            bot_response = chatbot.chat(message.strip())
            
            # Add to chat history
            chat_entry = {
                'user_message': message.strip(),
                'bot_response': bot_response,
                'timestamp': datetime.now().isoformat()
            }
            chat_history.append(chat_entry)
            
            return chat_history, ''
            
        except Exception as e:
            error_entry = {
                'user_message': message.strip(),
                'bot_response': f'Error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
            chat_history.append(error_entry)
            return chat_history, ''
    
    return chat_history, message or ''


@callback(
    Output('chat-container', 'children'),
    Input('chat-history-store', 'data')
)
def update_chat_display(chat_history):
    """Update the chat display with the current chat history."""
    if not chat_history:
        return html.Div([
            html.P(
                'Welcome! Start asking questions about the candidate\'s resume.',
                className='text-muted text-center',
                style={'margin-top': '50px'}
            )
        ])
    
    chat_messages = []
    for entry in chat_history:
        # User message
        chat_messages.append(
            create_message_component(entry['user_message'], is_user=True)
        )
        # Bot response
        chat_messages.append(
            create_message_component(entry['bot_response'], is_user=False)
        )
    
    return html.Div(chat_messages)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)