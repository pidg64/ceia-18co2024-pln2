"""
This script handles the parsing of a text resume, generating structured data,
and uploading it to a Pinecone vector database. It uses the Pinecone API
to create embeddings for semantic search.
"""

import os

from dotenv import load_dotenv
from typing import Dict, List, Any
from pinecone import Pinecone, CloudProvider, AwsRegion, IndexEmbed, EmbedModel

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
) -> None:
    if not pc.has_index(index_name):
        print(f"Creating index '{index_name}' with embedded model...")
        pc.create_index_for_model(
            name=index_name,
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1,
            embed=IndexEmbed(
                model=EmbedModel.Multilingual_E5_Large,
                field_map={"text": "chunk_text"},
                metric='cosine'
            )
        )
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")


def upload_chunks_via_records(pc: Pinecone, index_name: str, chunks: List[Dict[str, Any]], namespace: str = "default") -> None:
    """
    Uploads chunks using Pinecone's upsert_records with automatic embedding.

    Args:
        pc: Pinecone client
        index_name: Name of the index
        chunks: List of chunk dictionaries with 'id' and 'metadata'
        namespace: Pinecone namespace (default: 'default')
    """
    index_info = pc.describe_index(index_name)
    host = index_info.host
    index = pc.Index(host=host)
    records = []
    for chunk in chunks:
        if 'chunk_text' not in chunk['metadata']:
            continue
        record = {
            "_id": chunk["id"],
            "chunk_text": chunk["metadata"]["chunk_text"],
        }
        # Flatten metadata (excluding chunk_text)
        for key, value in chunk["metadata"].items():
            if key != "chunk_text":
                record[key] = value
        records.append(record)
    print(f"Uploading {len(records)} records to Pinecone...")
    index.upsert_records(namespace, records)
    print("Upload complete.")


def main():
    """
    Main function to execute the script logic.
    """
    try:
        load_api_key('PINECONE_API_KEY')
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
        create_index_if_needed(pc, index_name)
        print("Uploading chunks...")
        upload_chunks_via_records(pc, index_name, chunks, namespace="default")
        print("Done.")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
