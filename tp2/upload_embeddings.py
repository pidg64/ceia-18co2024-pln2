"""
This script handles the parsing of a text resume, generating structured data,
and uploading it to a Pinecone vector database. It uses the Pinecone API
to create embeddings for semantic search.
"""

import os
import hashlib

from dotenv import load_dotenv
from typing import List
from pinecone.db_data.index import Index
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStore
from pinecone import Pinecone, CloudProvider, AwsRegion, ServerlessSpec

from resume_management import parse_resume_to_chunks, read_resume_file


def load_api_key(key_name: str) -> str:
    """
    Loads an API key from environment variables.

    Args:
        key_name: The name of the environment variable.

    Returns:
        The API key as a string.

    Raises:
        ValueError: If the API key is not found in the environment.
    """
    load_dotenv()
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f'{key_name} not found in environment variables.')
    return api_key


def create_index_if_needed(
    pc: Pinecone,
    index_name: str,
) -> Index:
    """
    Creates a Pinecone index if it does not already exist.

    Args:
        pc: Pinecone client
        index_name: Name of the index

    Returns:
        The index instance
    """
    if not pc.has_index(index_name):
        print(f"Creating index '{index_name}' without embedded model...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1
            )
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
    return pc.Index(index_name)


def create_vector_store(
        index: Index,
        emb: Embeddings
    ) -> VectorStore:
    """
    Creates a vector store in Pinecone using the specified embeddings.

    Args:
        index: Pinecone Index instance
        emb: Embeddings instance to use for vectorization

    Returns:
        A Pinecone VectorStore instance
    
    Raises:
        ValueError: If the index instance is not provided.
    """
    if not index:
        raise ValueError("Index instance is required to create a vector store.")
    print(
        f"Creating vector store using {emb.__class__.__name__}"
    )
    return PineconeVectorStore(
        index=index,
        embedding=emb
    ) 


def upload_chunks_via_records(
    vector_store: VectorStore,
    chunks: List[Document]
    ) -> None:
    """
    Uploads chunks using Pinecone VectorStore.

    Args:
        vector_store: Pinecone VectorStore instance
        chunks: List of Document objects with 'page_content' and 'metadata'

    Returns:
        None
    """
    ids = [
        hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        for chunk in chunks
    ]
    print(f"Uploading {len(chunks)} chunks...")
    vector_store.add_documents(documents=chunks, ids=ids)


def main():
    """
    Main function to execute the script logic.
    """
    try:
        api_key = load_api_key('PINECONE_API_KEY')
        resume_path = os.path.join('cvs', 'dgpi_resume.txt')
        index_name = 'resume-index'
        print("Reading resume file...")
        resume_text = read_resume_file(resume_path)
        print("Parsing resume into chunks...")
        chunks = parse_resume_to_chunks(resume_text)
        if not chunks:
            print("No chunks were generated. Please check the resume file.")
            return
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        print("Ensuring index exists...")
        index = create_index_if_needed(pc, index_name)
        print("Creating embedding model...")
        emb = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768)
        print("Creating vector store...")
        vector_store = create_vector_store(index, emb)
        print("Uploading chunks...")
        upload_chunks_via_records(vector_store, chunks)
        print("Done.")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
