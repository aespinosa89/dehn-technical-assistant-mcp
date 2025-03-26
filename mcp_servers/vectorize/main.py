#!/usr/bin/env python3

import os
import json
import logging
import sqlite3
import hashlib
import asyncio
import numpy as np
import time
import httpx
from typing import Dict, Any, List, Optional, Union, Tuple
from mcp.server.fastmcp import FastMCP, Context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vectorize_mcp")

# Initialize MCP server
mcp = FastMCP("vectorize-mcp")

# SQLite database for vector storage
DB_PATH = os.getenv("VECTORIZE_DB_PATH", "vectorize.db")

# Database settings
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # Default to 384 for SentenceTransformers
DEFAULT_COLLECTION = "default"

# URL for embedding service
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8002/api/embeddings")

# Initialize database
def init_db():
    """Initialize the SQLite database for vector storage"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create collections table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS collections (
        name TEXT PRIMARY KEY,
        dimension INTEGER NOT NULL,
        metadata TEXT,
        created_at INTEGER
    )
    ''')
    
    # Create documents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        collection TEXT,
        content TEXT,
        metadata TEXT,
        embedding BLOB,  # Store binary embedding data
        created_at INTEGER,
        FOREIGN KEY (collection) REFERENCES collections(name)
    )
    ''')
    
    # Create index on collection
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents (collection)')
    
    # Insert default collection if it doesn't exist
    cursor.execute(
        'INSERT OR IGNORE INTO collections (name, dimension, metadata, created_at) VALUES (?, ?, ?, ?)',
        (DEFAULT_COLLECTION, EMBEDDING_DIMENSION, '{}', int(time.time()))
    )
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized at {DB_PATH}")

# Vector operations
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

async def get_embedding(text: str) -> List[float]:
    """
    Get embedding for text using the embedding service.
    If the service is unavailable, falls back to a mock embedding.
    """
    try:
        # Try to get embedding from service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                EMBEDDING_SERVICE_URL,
                json={"text": text},
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                return data["embedding"]
            else:
                logger.warning(f"Error from embedding service: {response.status_code}, {response.text}")
                # Fall back to mock embedding
                return await mock_embedding(text)
    except Exception as e:
        logger.warning(f"Could not reach embedding service: {e}")
        # Fall back to mock embedding
        return await mock_embedding(text)

async def mock_embedding(text: str) -> List[float]:
    """
    Generate a mock embedding for testing purposes.
    In production, this should be replaced with a real embedding model.
    """
    # Generate a deterministic but random-looking embedding based on the hash of the text
    np.random.seed(hash(text) % 2**32)
    embedding = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
    # Normalize the embedding to unit length
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

# Collection management functions
def create_collection(name: str, dimension: int = EMBEDDING_DIMENSION, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Create a new vector collection"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if collection already exists
        cursor.execute('SELECT name FROM collections WHERE name = ?', (name,))
        if cursor.fetchone():
            conn.close()
            return False
        
        # Create the collection
        cursor.execute(
            'INSERT INTO collections (name, dimension, metadata, created_at) VALUES (?, ?, ?, ?)',
            (name, dimension, json.dumps(metadata or {}), int(time.time()))
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error creating collection {name}: {e}")
        conn.close()
        return False

def get_collections() -> List[Dict[str, Any]]:
    """Get all collections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM collections')
    collections = [dict(row) for row in cursor.fetchall()]
    
    # Parse metadata
    for collection in collections:
        collection['metadata'] = json.loads(collection['metadata'])
    
    conn.close()
    return collections

def delete_collection(name: str) -> bool:
    """Delete a collection and all its documents"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if collection exists
        cursor.execute('SELECT name FROM collections WHERE name = ?', (name,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Delete all documents in the collection
        cursor.execute('DELETE FROM documents WHERE collection = ?', (name,))
        
        # Delete the collection
        cursor.execute('DELETE FROM collections WHERE name = ?', (name,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting collection {name}: {e}")
        conn.close()
        return False

# Document management functions
async def add_document(
    content: str,
    collection: str = DEFAULT_COLLECTION,
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None
) -> Optional[str]:
    """Add a document to a collection"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if collection exists
        cursor.execute('SELECT dimension FROM collections WHERE name = ?', (collection,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None
        
        dimension = result[0]
        
        # Generate embedding
        embedding = await get_embedding(content)
        
        # Ensure embedding dimension matches collection
        if len(embedding) != dimension:
            logger.error(f"Embedding dimension {len(embedding)} doesn't match collection dimension {dimension}")
            conn.close()
            return None
        
        # Generate ID if not provided
        if not document_id:
            document_id = hashlib.md5(f"{collection}:{content}:{time.time()}".encode()).hexdigest()
        
        # Store embedding as binary data
        embedding_binary = np.array(embedding, dtype=np.float32).tobytes()
        
        # Add document
        cursor.execute(
            '''
            INSERT OR REPLACE INTO documents (id, collection, content, metadata, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (document_id, collection, content, json.dumps(metadata or {}), embedding_binary, int(time.time()))
        )
        
        conn.commit()
        conn.close()
        return document_id
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        conn.close()
        return None

def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Get a document by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, collection, content, metadata, created_at FROM documents WHERE id = ?', (document_id,))
    document = cursor.fetchone()
    
    if not document:
        conn.close()
        return None
    
    document_dict = dict(document)
    document_dict['metadata'] = json.loads(document_dict['metadata'])
    
    conn.close()
    return document_dict

def delete_document(document_id: str) -> bool:
    """Delete a document by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        conn.close()
        return False

async def search_similar(
    query: str,
    collection: str = DEFAULT_COLLECTION,
    limit: int = 5,
    min_score: float = 0.7,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Search for similar documents using vector similarity"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Check if collection exists
        cursor.execute('SELECT dimension FROM collections WHERE name = ?', (collection,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return []
        
        # Generate query embedding
        query_embedding = await get_embedding(query)
        query_embedding_np = np.array(query_embedding)
        
        # Get all documents from the collection
        cursor.execute('SELECT id, collection, content, metadata, embedding, created_at FROM documents WHERE collection = ?', (collection,))
        documents = cursor.fetchall()
        
        # Calculate similarities and filter results
        results = []
        for doc in documents:
            doc_dict = dict(doc)
            doc_dict['metadata'] = json.loads(doc_dict['metadata'])
            
            # Check metadata filter if provided
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if key not in doc_dict['metadata'] or doc_dict['metadata'][key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Get embedding from binary data
            doc_embedding_binary = doc_dict.pop('embedding')
            doc_embedding_np = np.frombuffer(doc_embedding_binary, dtype=np.float32)
            
            # Calculate similarity
            similarity = cosine_similarity(query_embedding_np, doc_embedding_np)
            
            # Add to results if above threshold
            if similarity >= min_score:
                doc_dict['similarity'] = float(similarity)
                results.append(doc_dict)
        
        # Sort by similarity (descending) and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    except Exception as e:
        logger.error(f"Error searching similar documents: {e}")
        conn.close()
        return []
    finally:
        conn.close()
