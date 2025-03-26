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