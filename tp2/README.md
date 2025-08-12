# Resume RAG Chatbot

A web application that implements a **Retrieval-Augmented Generation (RAG)** chatbot for resume analysis. This system allows users to interact with an assistant that can answer questions about a candidate's resume using AI technologies.

## Features

- **RAG-powered Chat Interface**: Ask natural language questions about resume content
- **Real-time Progress Tracking**: Visual feedback during initialization with detailed progress steps
- **Async Chat Processing**: Non-blocking chat interactions with immediate message display
- **Multiple LLM Support**: Compatible with both OpenAI and Groq models
- **Vector Search**: Semantic search through resume content using Pinecone
- **Conversation Memory**: Maintains chat history for contextual conversations
- **Responsive Web UI**: Clean, modern interface built with Dash and Bootstrap

## Technology Stack

### Backend & AI
- **LangChain**: Framework for LLM applications and RAG implementation
- **Pinecone**: Cloud vector database for semantic search
- **OpenAI/Groq**: Large Language Model providers
- **OpenAI Embeddings**: Text vectorization for semantic search

### Web Framework
- **Dash**: Python web framework for building interactive applications
- **Dash Bootstrap Components**: UI components and styling

### Data Processing
- **LangChain Text Splitters**: Intelligent text chunking
- **Vector Embeddings**: Semantic text representation

## Project Structure

```
├── resume_chatbot_app.py      # Main Dash web application
├── llm.py                     # RAG chatbot implementation
├── upload_embeddings.py       # Resume processing and vector upload
├── resume_management.py       # Text processing utilities
├── requirements.txt           # Python dependencies
├── cvs/                       # Resume files directory
│   └── resume.txt             # Sample resume (John Doe's)
└── .env                       # Environment configuration
```

## Configuration

The application uses environment variables for configuration. Create a `.env` file in the project root:

```env
# API Keys (Required)
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
MODEL_TYPE=openai                           # Options: 'openai' or 'groq'
MODEL_NAME=gpt-5                            # Model name (e.g., gpt-5, llama3-8b-8192)
MODEL_TEMPERATURE=1                         # Model creativity (0.0-2.0)

# Data Configuration
RESUME_FILE_PATH=cvs/resume.txt             # Path to resume file
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=768
INDEX_NAME=resume-index

# Text Processing
DEFAULT_CHUNK_SIZE=625                      # Characters per chunk
DEFAULT_CHUNK_OVERLAP=125                   # Overlap between chunks
```

### Default Values

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `MODEL_TYPE` | `openai` | LLM provider (openai/groq) |
| `MODEL_NAME` | `gpt-5` | Specific model to use |
| `MODEL_TEMPERATURE` | `1` | Response creativity level |
| `RESUME_FILE_PATH` | `cvs/resume.txt` | Resume file location |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIMENSIONS` | `768` | Vector embedding size |
| `INDEX_NAME` | `resume-index` | Pinecone index name |
| `DEFAULT_CHUNK_SIZE` | `625` | Text chunk size in characters |
| `DEFAULT_CHUNK_OVERLAP` | `125` | Overlap between chunks |

## Installation & Setup

### Prerequisites

1. **Python 3.10+** installed
2. **API Keys** for:
   - Pinecone (vector database)
   - OpenAI (for embeddings and optionally LLM)
   - Groq (optional, for faster inference)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tp2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env  # Copy the sample configuration
   # Edit .env with your API keys and preferences
   ```

5. **Prepare resume data**
   - Place resume text files in the `cvs/` directory
   - Update `RESUME_FILE_PATH` in `.env` if needed

## Usage

### Starting the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the application
python resume_chatbot_app.py
```

The application will be available at: `http://localhost:8050`

### Using the Chat Interface

1. **Initialization**: The app automatically initializes when started
   - Connects to Pinecone
   - Processes resume text into chunks
   - Creates vector embeddings
   - Shows real-time progress

2. **Chat Interaction**: 
   - Type questions about the candidate's experience, skills, or education
   - Press Enter or click "Send"
   - Messages appear immediately
   - Bot responses show "Processing data..." while generating

3. **Sample Questions**:
   - "Where did John Doe work between January 2023 to August 2025?"
   - "What responsibilities did he have in Microsoft?"
   - "What certifications does the candidate have?"
   - "Describe the candidate educational background"

## How It Works

### RAG Architecture

1. **Document Processing**: 
   - Resume text is split into overlapping chunks
   - Each chunk is converted to vector embeddings
   - Stored in Pinecone vector database

2. **Query Processing**:
   - User question is converted to vector embedding
   - Semantic search finds relevant resume chunks
   - Context is retrieved based on similarity scores

3. **Response Generation**:
   - Retrieved context + conversation history sent to LLM
   - AI generates natural, contextual responses
   - Maintains conversation memory for follow-up questions

### Key Components

- **`ResumeRAGChatbot`**: Core chatbot logic with RAG implementation
- **`initialize_chatbot()`**: Sets up vector store and AI models
- **Progress Monitoring**: Real-time UI updates during initialization
- **Async Processing**: Non-blocking chat interactions

## Advanced Features

### Conversation Memory
- Maintains recent conversation history
- Enables contextual follow-up questions
- Configurable history length

### Semantic Search
- Uses cosine similarity for document retrieval
- Configurable similarity thresholds
- Returns multiple relevant chunks with scores

### Multi-Model Support
- Switch between OpenAI and Groq models
- Easy model configuration via environment variables
- Support for different temperature settings

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in `.env`
2. **Model Not Found**: Check `MODEL_NAME` matches available models
3. **Pinecone Connection**: Verify `PINECONE_API_KEY` and internet connection
4. **Resume File**: Ensure `RESUME_FILE_PATH` points to valid text file

## Customization

### Adding New Resume
1. Place resume text file in `cvs/` directory
2. Update `RESUME_FILE_PATH` in `.env`
3. Restart the application

### Changing Models
Update `.env` configuration:
```env
MODEL_TYPE=groq
MODEL_NAME=llama3-8b-8192
```

### Adjusting Chunk Size
Modify text processing parameters:
```env
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
```