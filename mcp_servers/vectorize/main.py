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
