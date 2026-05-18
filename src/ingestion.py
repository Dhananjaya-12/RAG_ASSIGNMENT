"""
Document Ingestion Module
Handles loading documents from disk and chunking them into smaller pieces
for embedding and indexing.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self):
        preview = self.text[:80].replace('\n', ' ')
        return f"Chunk({self.source_file}#{self.chunk_index}: '{preview}...')"


def load_documents(directory: str, extensions: Optional[List[str]] = None) -> List[Dict]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to the directory containing documents
        extensions: List of file extensions to load (e.g., ['.txt', '.md'])
    
    Returns:
        List of dicts with 'text', 'source', and 'metadata' keys
    """
    if extensions is None:
        extensions = [".txt", ".md"]
    
    documents = []
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")
    
    for filepath in sorted(directory.rglob("*")):
        if filepath.suffix.lower() in extensions and filepath.is_file():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if content:  # Skip empty files
                    documents.append({
                        "text": content,
                        "source": str(filepath.name),
                        "metadata": {
                            "full_path": str(filepath),
                            "file_size": filepath.stat().st_size,
                            "extension": filepath.suffix,
                        }
                    })
            except Exception as e:
                print(f"  [WARNING] Could not read {filepath}: {e}")
    
    return documents


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """
    Split documents into overlapping chunks using a sliding window approach.
    
    Chunks are split at paragraph or sentence boundaries when possible
    to preserve semantic coherence.
    
    Args:
        documents: List of document dicts from load_documents()
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between consecutive chunks
    
    Returns:
        List of DocumentChunk objects
    """
    all_chunks = []
    
    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        metadata = doc.get("metadata", {})
        
        # Split into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Accumulate paragraphs into chunks
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        char_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                char_pos += 1
                continue
            
            # If adding this paragraph exceeds chunk_size and we have content
            if len(current_chunk) + len(para) + 1 > chunk_size and current_chunk:
                # Save current chunk
                chunk = DocumentChunk(
                    text=current_chunk.strip(),
                    source_file=source,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata=metadata.copy(),
                )
                all_chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                # Take the last `chunk_overlap` characters from current chunk
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_start = current_start + len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = char_pos
            
            char_pos += len(para) + 2  # +2 for the paragraph separator
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                source_file=source,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata=metadata.copy(),
            )
            all_chunks.append(chunk)
    
    return all_chunks


def get_chunk_texts(chunks: List[DocumentChunk]) -> List[str]:
    """Extract just the text content from a list of chunks."""
    return [chunk.text for chunk in chunks]


def get_chunk_metadata(chunks: List[DocumentChunk]) -> List[Dict]:
    """Extract serializable metadata from chunks for storage."""
    return [
        {
            "text": chunk.text,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
