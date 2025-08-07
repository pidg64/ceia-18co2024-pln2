from typing import Any, Dict, List


def read_resume_file(file_path: str) -> str:
    """
    Reads the content of the resume text file.

    Args:
        file_path: The path to the resume file.

    Returns:
        The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_resume_to_chunks(
    resume_text: str, chunk_size: int = 1000, overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Parses the resume text and converts it into a list of overlapping chunks.

    Args:
        resume_text: The full text of the resume.
        chunk_size: The size of each chunk in characters.
        overlap: The overlap between consecutive chunks in characters.

    Returns:
        A list of dictionaries, where each dictionary is a chunk to be embedded.
    """
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap")
    chunks = []
    start_index = 0
    chunk_id = 1
    while start_index < len(resume_text):
        end_index = start_index + chunk_size
        chunk_text = resume_text[start_index:end_index]
        chunks.append({
            'id': f'chunk_{chunk_id}',
            'metadata': {
                'section': 'resume_text',
                'chunk_text': chunk_text
            }
        })
        start_index += chunk_size - overlap
        chunk_id += 1 
    return chunks