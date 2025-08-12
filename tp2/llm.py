"""
This module handles LLM integration for chat functionality using LangChain with
RAG capabilities.
"""

import textwrap

from typing import List
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel


# Create the prompt template with system message and conversation history
system_prompt = textwrap.dedent(
    """
    You are a helpful assistant that answers questions about a candidate's 
    resume based on the provided context.

    Rules:
    - If CONTEXT is non-empty, use it. If CONTEXT is empty, you must answer 
    from conversation history. Only say you do not know if neither has the 
    information.
    - Respond in a friendly and conversational manner, as if you're having 
    a natural conversation about the candidate
    - Use clear, direct language and avoid overly formal or robotic responses
    - When discussing experience, mention specific companies, roles, and 
    timeframes
    - For skills questions, be specific about the technologies mentioned

    CONTEXT:
    {context}
    """
)

prompt_template = ChatPromptTemplate([
    ('system', system_prompt),
    MessagesPlaceholder('msgs')
])


class ResumeRAGChatbot:
    """
    A chatbot that uses Groq LLM with RAG to answer questions about a resume.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        model: BaseChatModel,
        max_history_length: int = 5
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            vector_store: Pinecone vector store containing resume data
            model: Chat model instance (ChatGroq or ChatOpenAI)
            max_history_length: Maximum number of past message pairs to retain
        """
        self.vector_store = vector_store
        self.llm = model
        self.max_history_length = max_history_length
        self.message_history: List[HumanMessage | AIMessage] = []
    
    def _retrieve_context(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.35
    ) -> str:
        """
        Retrieve relevant context from the vector store.
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            score_threshold: Minimum score to consider a document relevant
            
        Returns:
            Concatenated context from retrieved documents
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            filtered_results = [
                (doc, score) for doc, score in results 
                if score > score_threshold
            ]
            
            if not filtered_results:
                return ''
            
            context_parts = []
            for doc, score in filtered_results:
                context_parts.append(
                    f'[Score: {score:.3f}] {doc.page_content}'
                )
            
            return '\n\n'.join(context_parts)
            
        except Exception as e:
            return f'Error retrieving context: {str(e)}'
    
    def _build_messages_for_prompt(
        self, 
        current_human_message: str
    ) -> List[HumanMessage | AIMessage]:
        """
        Build the messages list for the prompt template.
        
        Args:
            current_human_message: Current user message
            
        Returns:
            List of messages including conversation history and current message
        """
        messages = []
        
        # Add recent conversation history (last interaction)
        if len(self.message_history) >= 2:
            # Get the last human-AI interaction
            last_human = self.message_history[-2]
            last_ai = self.message_history[-1]
            messages.extend([last_human, last_ai])
        
        # Add current human message
        messages.append(HumanMessage(content=current_human_message))
        
        return messages
    
    def _update_message_history(
        self, 
        human_message: HumanMessage,
        ai_message: AIMessage
    ) -> None:
        """
        Update the message history with new messages.
        
        Args:
            human_message: The user's message
            ai_message: The AI's response
        """
        self.message_history.extend([human_message, ai_message])
        
        # Keep only recent messages to prevent context overflow
        if len(self.message_history) > self.max_history_length * 2:
            # Remove oldest pair (human + AI message)
            self.message_history = self.message_history[2:]
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response using RAG.
        
        Args:
            user_message: The user's question
            
        Returns:
            The chatbot's response
        """
        try:
            # Retrieve relevant context for current question
            context = self._retrieve_context(user_message)
            print('Retrieved Context'.center(50, '-'))
            print(context)
            print('End of Context'.center(50, '-'))
            
            # Build messages for prompt (history + current)
            messages_to_pass = self._build_messages_for_prompt(user_message)
            
            # Format the prompt with context and messages
            formatted_prompt = prompt_template.invoke({
                'context': context, 
                'msgs': messages_to_pass
            })
            
            print(
                f'Conversation has {len(self.message_history) // 2} '
                f'previous exchanges'
            )
            print('Formatted Prompt:'.center(50, '-'))
            print(formatted_prompt)
            print('End of Formatted Prompt'.center(50, '-'))
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            
            # Extract text content and create AI message
            if hasattr(response, 'content'):
                bot_response = str(response.content)
            else:
                bot_response = str(response)
            
            print(f'LLM response: {bot_response}')
            
            # Create message objects for history
            human_message = HumanMessage(content=user_message)
            ai_message = AIMessage(content=bot_response)
            
            # Update conversation history
            self._update_message_history(human_message, ai_message)
            
            return bot_response
            
        except Exception as e:
            error_msg = f'Error processing your message: {str(e)}'
            return error_msg
    
    def get_message_history(self) -> List[HumanMessage | AIMessage]:
        """
        Get the current message history with proper message types.
        
        Returns:
            List of HumanMessage and AIMessage objects
        """
        return self.message_history.copy()
    
    def clear_history(self) -> None:
        """Clear message history."""
        self.message_history.clear()
    
    def get_conversation_context_length(self) -> int:
        """
        Get the current length of the conversation context.
        
        Returns:
            Number of message pairs in history
        """
        return len(self.message_history) // 2