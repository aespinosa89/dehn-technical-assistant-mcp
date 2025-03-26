#!/usr/bin/env python3

import os
import json
import asyncio
import logging
import psycopg2
import psycopg2.extras
from typing import Dict, Any, List, Optional, Tuple
from mcp.server.fastmcp import FastMCP, Context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("postgres_mcp")

# Initialize MCP server
mcp = FastMCP("postgres-mcp")

# Database connection settings from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/dehn")

# Global connection pool
conn_pool = None

# Helper functions
def get_connection():
    """Get a connection from the pool"""
    global conn_pool
    if conn_pool is None:
        try:
            conn_pool = psycopg2.connect(DATABASE_URL)
            conn_pool.autocommit = True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    return conn_pool

def get_tables() -> List[str]:
    """Get a list of all tables in the database"""
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        return [row[0] for row in cursor.fetchall()]

def get_table_schema(table_name: str) -> str:
    """Get the schema for a specific table"""
    conn = get_connection()
    with conn.cursor() as cursor:
        # Get column information
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        columns = cursor.fetchall()
        
        # Get primary key information
        cursor.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
        """, (table_name,))
        pk_columns = [row[0] for row in cursor.fetchall()]
        
        # Format the schema
        schema = [f"Table: {table_name}\n"]
        schema.append("Columns:")
        for col in columns:
            col_name, data_type, is_nullable, default = col
            pk_indicator = " (PK)" if col_name in pk_columns else ""
            nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
            default_str = f" DEFAULT {default}" if default else ""
            schema.append(f"  {col_name}{pk_indicator}: {data_type} {nullable}{default_str}")
        
        # Get foreign key information
        cursor.execute("""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s
        """, (table_name,))
        fk_info = cursor.fetchall()
        
        if fk_info:
            schema.append("\nForeign Keys:")
            for fk in fk_info:
                col_name, foreign_table, foreign_col = fk
                schema.append(f"  {col_name} -> {foreign_table}.{foreign_col}")
        
        return "\n".join(schema)

def execute_query(query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """Execute a SQL query and return the results as a list of dictionaries"""
    conn = get_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        try:
            cursor.execute(query, params)
            
            # Check if it's a SELECT query by checking if it has a result set
            if cursor.description is not None:
                return cursor.fetchall()
            else:
                # For non-SELECT queries, return row count
                return [{"rows_affected": cursor.rowcount}]
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

# Resources
@mcp.resource("schema://tables")
def get_all_tables() -> str:
    """Get a list of all tables in the database"""
    try:
        tables = get_tables()
        return "Available Tables:\n" + "\n".join(tables)
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return f"Error getting tables: {str(e)}"

@mcp.resource("schema://table/{table_name}")
def get_table_schema_resource(table_name: str) -> str:
    """Get the schema for a specific table"""
    try:
        schema = get_table_schema(table_name)
        return schema
    except Exception as e:
        logger.error(f"Error getting schema for table {table_name}: {e}")
        return f"Error getting schema for table {table_name}: {str(e)}"

# Tools
@mcp.tool()
async def execute_select(query: str) -> str:
    """Execute a read-only SELECT SQL query.
    
    Args:
        query: A SELECT SQL query string
    """
    # Security check: only allow SELECT queries
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for security reasons."
    
    try:
        results = execute_query(query)
        
        # Format results as a table
        if not results:
            return "Query executed successfully. No results returned."
        
        # Get column names from the first result
        columns = results[0].keys()
        
        # Format header
        header = " | ".join(columns)
        separator = "-+-".join(["-" * len(col) for col in columns])
        
        # Format rows
        rows = []
        for row in results:
            formatted_row = " | ".join([str(row[col]) for col in columns])
            rows.append(formatted_row)
        
        # Combine everything
        table = f"{header}\n{separator}\n" + "\n".join(rows)
        
        return f"Query executed successfully. Results:\n\n{table}"
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return f"Error executing query: {str(e)}"

@mcp.tool()
async def get_table_info(table_name: str) -> str:
    """Get detailed information about a table including schema and sample data.
    
    Args:
        table_name: The name of the table to analyze
    """
    try:
        # Get schema
        schema = get_table_schema(table_name)
        
        # Get sample data (up to 5 rows)
        sample_data = execute_query(f"SELECT * FROM {table_name} LIMIT 5")
        
        # Format sample data
        if sample_data:
            columns = sample_data[0].keys()
            header = " | ".join(columns)
            separator = "-+-".join(["-" * len(col) for col in columns])
            
            rows = []
            for row in sample_data:
                formatted_row = " | ".join([str(row[col]) for col in columns])
                rows.append(formatted_row)
            
            formatted_sample = f"{header}\n{separator}\n" + "\n".join(rows)
        else:
            formatted_sample = "No data in table."
        
        # Get row count
        row_count = execute_query(f"SELECT COUNT(*) as count FROM {table_name}")[0]["count"]
        
        return f"{schema}\n\nRow Count: {row_count}\n\nSample Data:\n{formatted_sample}"
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        return f"Error getting table info: {str(e)}"

@mcp.tool()
async def describe_database() -> str:
    """Get an overview of the database structure including all tables and their relationships."""
    try:
        tables = get_tables()
        
        # For each table, get schema and relationship info
        descriptions = []
        relationships = []
        
        for table in tables:
            # Get row count
            row_count = execute_query(f"SELECT COUNT(*) as count FROM {table}")[0]["count"]
            
            # Get primary keys
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                        AND tc.table_schema = 'public'
                        AND tc.table_name = %s
                """, (table,))
                pk_columns = [row[0] for row in cursor.fetchall()]
                pk_str = ", ".join(pk_columns) if pk_columns else "None"
            
            # Add table description
            descriptions.append(f"- {table}: {row_count} rows, PK: {pk_str}")
            
            # Get relationships
            conn = get_connection()
            with conn.cursor() as cursor:
                # Foreign keys from this table
                cursor.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_schema = 'public'
                        AND tc.table_name = %s
                """, (table,))
                fk_out = cursor.fetchall()
                
                for fk in fk_out:
                    col_name, foreign_table, foreign_col = fk
                    relationships.append(f"- {table}.{col_name} → {foreign_table}.{foreign_col}")
        
        # Compile the overview
        overview = "Database Overview:\n\n"
        overview += "Tables:\n" + "\n".join(descriptions)
        
        if relationships:
            overview += "\n\nRelationships:\n" + "\n".join(relationships)
        
        return overview
    except Exception as e:
        logger.error(f"Error describing database: {e}")
        return f"Error describing database: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
