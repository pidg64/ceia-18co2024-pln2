"""
This script provides functionality to query a Pinecone vector database
and optionally upload resume chunks for semantic search capabilities.
"""

import os
import argparse

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from upload_embeddings import (
    load_api_key,
    create_index_if_needed,
    create_vector_store,
    upload_chunks_via_records,
    read_resume_file,
    parse_resume_to_chunks
)

def upload_resume_chunks(vector_store):
    """
    Function to upload resume chunks to Pinecone.
    
    Args:
        vector_store: The Pinecone vector store instance
    """
    try:
        resume_path = os.path.join('cvs', 'dgpi_resume.txt')
        print("Reading resume file...")
        resume_text = read_resume_file(resume_path)
        print("Parsing resume into chunks...")
        chunks = parse_resume_to_chunks(resume_text, chunk_size=625, overlap=125)
        if not chunks:
            print("No chunks were generated. Please check the resume file.")
            return
        print("Uploading chunks...")
        upload_chunks_via_records(vector_store, chunks)
        print("Upload completed.")
    except Exception as e:
        print(f"Error uploading chunks: {e}")


def query_vector_store(vector_store, query: str, k: int = 3):
    """
    Query the vector store and display results with debugging info.
    
    Args:
        vector_store: The Pinecone vector store instance
        query: The search query string
        k: Number of results to return
    """
    try:
        print(f"Querying for: '{query}'")
        print(f"Requesting {k} similar chunks...")        
        # Check if there are any documents in the index
        index_stats = vector_store._index.describe_index_stats()
        print(f"Index stats: {index_stats}")        
        results_with_scores = vector_store.similarity_search_with_score(
            query, k=k
        )        
        return results_with_scores              
    except Exception as e:
        print(f"Error during query: {e}")


def main():
    """
    Main function to execute the script logic.
    """
    parser = argparse.ArgumentParser(
        description='Query Pinecone or upload embeddings'
    )
    parser.add_argument(
        '-u', '--upload', 
        action='store_true',
        help='Upload embeddings to Pinecone'
    )
    parser.add_argument(
        '-q', '--query',
        type=str,
        default='What job experiences does the candidate have?',
        help=(
            'Query string to search for (default: What job experiences does '
            'the candidate have?)'
        )
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=3,
        help='Number of top results to return (default: 3)'
    )
    args = parser.parse_args()
    try:
        api_key = load_api_key('PINECONE_API_KEY')
        index_name = 'resume-index'            
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        print("Ensuring index exists...")
        index = create_index_if_needed(pc, index_name)
        print("Creating embedding model...")
        emb = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768)
        print("Creating vector store...")
        vector_store = create_vector_store(index, emb)
        if args.upload:
            upload_resume_chunks(vector_store)
        else:
            results = query_vector_store(vector_store, args.query, args.top_k)
            if not results:
                print("No results found.")
                return
            else:
                print(f"Found {len(results)} results:")
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\nChunk {i} (Score: {score:.4f}):")
                    print(f"'''\n{doc.page_content}\n'''")
                    print(f"Metadata: {doc.metadata}")
                    print("-" * 40)  
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()