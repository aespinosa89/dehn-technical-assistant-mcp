#!/usr/bin/env python3

import os
import json
import time
import asyncio
import logging
import sqlite3
import hashlib
import datetime
from typing import Dict, Any, List, Optional, Union
from mcp.server.fastmcp import FastMCP, Context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_mcp")

# Initialize MCP server
mcp = FastMCP("memory-mcp")

# SQLite database for memory storage
DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")

# Memory types
MEMORY_TYPES = {
    "conversation": "Conversation history",
    "fact": "Factual information",
    "user_preference": "User preferences",
    "product_interaction": "Product interaction history"
}

# Initialize database
def init_db():
    """Initialize the SQLite database for memory storage"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create memory table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memory (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        content TEXT,
        metadata TEXT,
        memory_type TEXT,
        created_at INTEGER,
        accessed_at INTEGER,
        access_count INTEGER DEFAULT 0
    )
    ''')
    
    # Create index for searching
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_memory_type ON memory (user_id, memory_type)')
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized at {DB_PATH}")

# Memory management functions
def store_memory(
    user_id: str,
    content: str,
    memory_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Store a memory in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Generate a unique ID
    memory_id = hashlib.md5(f"{user_id}:{content}:{time.time()}".encode()).hexdigest()
    
    # Current timestamp
    now = int(time.time())
    
    # Insert the memory
    cursor.execute(
        '''
        INSERT INTO memory (id, user_id, content, metadata, memory_type, created_at, accessed_at, access_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (memory_id, user_id, content, json.dumps(metadata or {}), memory_type, now, now, 0)
    )
    
    conn.commit()
    conn.close()
    
    return memory_id

def get_memory(memory_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific memory by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Retrieve the memory
    cursor.execute('SELECT * FROM memory WHERE id = ?', (memory_id,))
    memory = cursor.fetchone()
    
    if memory:
        # Update access information
        now = int(time.time())
        cursor.execute(
            'UPDATE memory SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?',
            (now, memory_id)
        )
        conn.commit()
        
        # Convert to dict and parse metadata
        memory_dict = dict(memory)
        memory_dict['metadata'] = json.loads(memory_dict['metadata'])
        
        conn.close()
        return memory_dict
    
    conn.close()
    return None

def retrieve_memories(
    user_id: str,
    memory_type: Optional[str] = None,
    limit: int = 10,
    order_by: str = "created_at",
    filter_metadata: Optional[Dict[str, Any]] = None,
    search_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve memories based on various filters"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Build the query
    query = 'SELECT * FROM memory WHERE user_id = ?'
    params = [user_id]
    
    if memory_type:
        query += ' AND memory_type = ?'
        params.append(memory_type)
    
    if search_query:
        query += ' AND content LIKE ?'
        params.append(f'%{search_query}%')
    
    # Add order by clause
    if order_by in ['created_at', 'accessed_at', 'access_count']:
        query += f' ORDER BY {order_by} DESC'
    else:
        query += ' ORDER BY created_at DESC'
    
    # Add limit
    query += ' LIMIT ?'
    params.append(limit)
    
    # Execute the query
    cursor.execute(query, params)
    memories = cursor.fetchall()
    
    # Process the results
    result = []
    for memory in memories:
        memory_dict = dict(memory)
        memory_dict['metadata'] = json.loads(memory_dict['metadata'])
        
        # Apply metadata filter if specified
        if filter_metadata:
            include = True
            for key, value in filter_metadata.items():
                if key not in memory_dict['metadata'] or memory_dict['metadata'][key] != value:
                    include = False
                    break
            if not include:
                continue
        
        result.append(memory_dict)
    
    conn.close()
    return result

def delete_memory(memory_id: str) -> bool:
    """Delete a memory by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM memory WHERE id = ?', (memory_id,))
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    
    return deleted

def update_memory(memory_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Update a memory's content or metadata"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First check if the memory exists
    cursor.execute('SELECT metadata FROM memory WHERE id = ?', (memory_id,))
    existing = cursor.fetchone()
    
    if not existing:
        conn.close()
        return False
    
    # Prepare the update
    update_parts = []
    params = []
    
    if content is not None:
        update_parts.append('content = ?')
        params.append(content)
    
    if metadata is not None:
        existing_metadata = json.loads(existing[0])
        # Merge existing metadata with new metadata
        updated_metadata = {**existing_metadata, **metadata}
        update_parts.append('metadata = ?')
        params.append(json.dumps(updated_metadata))
    
    if not update_parts:
        conn.close()
        return True  # Nothing to update
    
    # Update the memory
    params.append(memory_id)
    cursor.execute(
        f'UPDATE memory SET {", ".join(update_parts)} WHERE id = ?',
        params
    )
    
    conn.commit()
    conn.close()
    
    return True

# Tools
@mcp.tool()
async def store_conversation_memory(user_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Store a conversation message in memory.
    
    Args:
        user_id: Unique identifier for the user
        message: The conversation message to store
        metadata: Optional metadata about the message (e.g., timestamp, sender, context)
    """
    try:
        init_db()
        memory_id = store_memory(
            user_id=user_id,
            content=message,
            memory_type="conversation",
            metadata=metadata
        )
        
        return f"Memory stored successfully with ID: {memory_id}"
    except Exception as e:
        logger.error(f"Error storing conversation memory: {e}")
        return f"Error storing memory: {str(e)}"

@mcp.tool()
async def retrieve_conversation_history(user_id: str, limit: int = 10, search_query: Optional[str] = None) -> str:
    """Retrieve conversation history for a user.
    
    Args:
        user_id: Unique identifier for the user
        limit: Maximum number of messages to retrieve (default: 10)
        search_query: Optional search term to filter messages
    """
    try:
        init_db()
        memories = retrieve_memories(
            user_id=user_id,
            memory_type="conversation",
            limit=limit,
            order_by="created_at",
            search_query=search_query
        )
        
        if not memories:
            return f"No conversation history found for user {user_id}"
        
        # Format the conversation history
        formatted = []
        for memory in memories:
            created_time = datetime.datetime.fromtimestamp(memory['created_at']).strftime('%Y-%m-%d %H:%M:%S')
            role = memory['metadata'].get('role', 'unknown')
            formatted.append(f"[{created_time}] {role}: {memory['content']}")
        
        return "Conversation History:\n\n" + "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return f"Error retrieving conversation history: {str(e)}"

@mcp.tool()
async def store_user_preference(user_id: str, preference_key: str, preference_value: str) -> str:
    """Store a user preference.
    
    Args:
        user_id: Unique identifier for the user
        preference_key: The preference name/key
        preference_value: The preference value
    """
    try:
        init_db()
        # Check if preference already exists
        memories = retrieve_memories(
            user_id=user_id,
            memory_type="user_preference",
            filter_metadata={"preference_key": preference_key}
        )
        
        metadata = {
            "preference_key": preference_key,
            "updated_at": int(time.time())
        }
        
        if memories:
            # Update existing preference
            memory_id = memories[0]['id']
            update_memory(memory_id, content=preference_value, metadata=metadata)
            return f"User preference '{preference_key}' updated successfully"
        else:
            # Store new preference
            memory_id = store_memory(
                user_id=user_id,
                content=preference_value,
                memory_type="user_preference",
                metadata=metadata
            )
            return f"User preference '{preference_key}' stored successfully"
    except Exception as e:
        logger.error(f"Error storing user preference: {e}")
        return f"Error storing user preference: {str(e)}"

@mcp.tool()
async def get_user_preferences(user_id: str) -> str:
    """Get all preferences for a user.
    
    Args:
        user_id: Unique identifier for the user
    """
    try:
        init_db()
        memories = retrieve_memories(
            user_id=user_id,
            memory_type="user_preference",
            limit=100
        )
        
        if not memories:
            return f"No preferences found for user {user_id}"
        
        # Format the preferences
        formatted = []
        for memory in memories:
            key = memory['metadata'].get('preference_key', 'unknown')
            formatted.append(f"{key}: {memory['content']}")
        
        return "User Preferences:\n\n" + "\n".join(formatted)
    except Exception as e:
        logger.error(f"Error retrieving user preferences: {e}")
        return f"Error retrieving user preferences: {str(e)}"

@mcp.tool()
async def store_product_interaction(
    user_id: str,
    product_id: str,
    interaction_type: str,
    details: Optional[str] = None
) -> str:
    """Store a product interaction for a user.
    
    Args:
        user_id: Unique identifier for the user
        product_id: Identifier for the product
        interaction_type: Type of interaction (e.g., view, purchase, question)
        details: Optional details about the interaction
    """
    try:
        init_db()
        metadata = {
            "product_id": product_id,
            "interaction_type": interaction_type,
            "timestamp": int(time.time())
        }
        
        content = details or f"{interaction_type} interaction with product {product_id}"
        
        memory_id = store_memory(
            user_id=user_id,
            content=content,
            memory_type="product_interaction",
            metadata=metadata
        )
        
        return f"Product interaction stored successfully with ID: {memory_id}"
    except Exception as e:
        logger.error(f"Error storing product interaction: {e}")
        return f"Error storing product interaction: {str(e)}"

@mcp.tool()
async def get_product_interactions(user_id: str, product_id: Optional[str] = None) -> str:
    """Get product interactions for a user.
    
    Args:
        user_id: Unique identifier for the user
        product_id: Optional product ID to filter interactions
    """
    try:
        init_db()
        filter_metadata = {"product_id": product_id} if product_id else None
        
        memories = retrieve_memories(
            user_id=user_id,
            memory_type="product_interaction",
            limit=50,
            filter_metadata=filter_metadata
        )
        
        if not memories:
            return f"No product interactions found for user {user_id}"
        
        # Format the interactions
        formatted = []
        for memory in memories:
            timestamp = datetime.datetime.fromtimestamp(memory['metadata'].get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
            product_id = memory['metadata'].get('product_id', 'unknown')
            interaction_type = memory['metadata'].get('interaction_type', 'unknown')
            
            formatted.append(f"[{timestamp}] {interaction_type} - Product: {product_id} - {memory['content']}")
        
        return "Product Interactions:\n\n" + "\n".join(formatted)
    except Exception as e:
        logger.error(f"Error retrieving product interactions: {e}")
        return f"Error retrieving product interactions: {str(e)}"

if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Run the MCP server
    mcp.run()
