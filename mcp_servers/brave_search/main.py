#!/usr/bin/env python3

import os
import json
import httpx
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from mcp.server.fastmcp import FastMCP, Context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brave_search_mcp")

# Initialize MCP server
mcp = FastMCP("brave-search-mcp")

# API configuration
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1"
BRAVE_NEWS_API_URL = "https://api.search.brave.com/news/v1"

if not BRAVE_API_KEY:
    logger.error("BRAVE_API_KEY environment variable is required")

# Helper functions
async def perform_web_search(
    query: str,
    count: int = 10,
    offset: int = 0,
    search_filter: Optional[str] = None,
    country: Optional[str] = None
) -> Dict[str, Any]:
    """Perform a web search using Brave Search API"""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    
    params = {
        "q": query,
        "count": min(count, 20),  # Max 20 results per request
        "offset": offset
    }
    
    if search_filter:
        params["search_filter"] = search_filter
        
    if country:
        params["country"] = country
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BRAVE_SEARCH_API_URL}/search",
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return {"error": str(e)}

async def perform_local_search(
    query: str,
    count: int = 5,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """Perform a local search using Brave Local API"""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    
    params = {
        "q": query,
        "count": min(count, 20)  # Max 20 results per request
    }
    
    if location:
        params["location"] = location
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BRAVE_SEARCH_API_URL}/local",
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error performing local search: {e}")
        return {"error": str(e)}

async def perform_news_search(
    query: str,
    count: int = 10,
    country: Optional[str] = None,
    freshness: Optional[str] = None
) -> Dict[str, Any]:
    """Perform a news search using Brave News API"""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    
    params = {
        "q": query,
        "count": min(count, 20)  # Max 20 results per request
    }
    
    if country:
        params["country"] = country
        
    if freshness:
        params["freshness"] = freshness
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BRAVE_NEWS_API_URL}/search",
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error performing news search: {e}")
        return {"error": str(e)}

def format_web_search_results(results: Dict[str, Any]) -> str:
    """Format web search results in a readable format"""
    if "error" in results:
        return f"Error: {results['error']}"
    
    if "web" not in results or not results["web"].get("results"):
        return "No web search results found."
    
    # Extract the most relevant information
    formatted = ["Web Search Results:"]
    
    for idx, result in enumerate(results["web"]["results"], 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        description = result.get("description", "No description")
        
        formatted.append(f"{idx}. {title}")
        formatted.append(f"   URL: {url}")
        formatted.append(f"   {description}")
        formatted.append("")
    
    if "query" in results:
        formatted.append(f"Search Query: {results['query']}")
        
    return "\n".join(formatted)

def format_local_search_results(results: Dict[str, Any]) -> str:
    """Format local search results in a readable format"""
    if "error" in results:
        return f"Error: {results['error']}"
    
    if "local" not in results or not results["local"].get("places"):
        return "No local search results found."
    
    # Extract the most relevant information
    formatted = ["Local Search Results:"]
    
    for idx, place in enumerate(results["local"]["places"], 1):
        name = place.get("name", "No name")
        address = place.get("full_address", "No address")
        category = place.get("category", "")
        phone = place.get("phone", "")
        rating = place.get("rating", {}).get("value", "No rating")
        review_count = place.get("rating", {}).get("count", 0)
        
        formatted.append(f"{idx}. {name}")
        if category:
            formatted.append(f"   Category: {category}")
        formatted.append(f"   Address: {address}")
        if phone:
            formatted.append(f"   Phone: {phone}")
        if rating != "No rating":
            formatted.append(f"   Rating: {rating}/5 ({review_count} reviews)")
        formatted.append("")
    
    if "query" in results:
        formatted.append(f"Search Query: {results['query']}")
        
    return "\n".join(formatted)

def format_news_search_results(results: Dict[str, Any]) -> str:
    """Format news search results in a readable format"""
    if "error" in results:
        return f"Error: {results['error']}"
    
    if "news" not in results or not results["news"].get("results"):
        return "No news search results found."
    
    # Extract the most relevant information
    formatted = ["News Search Results:"]
    
    for idx, result in enumerate(results["news"]["results"], 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        description = result.get("description", "No description")
        source = result.get("source", "Unknown source")
        published_time = result.get("published_time", "")
        
        formatted.append(f"{idx}. {title}")
        formatted.append(f"   Source: {source}")
        if published_time:
            formatted.append(f"   Published: {published_time}")
        formatted.append(f"   URL: {url}")
        formatted.append(f"   {description}")
        formatted.append("")
    
    if "query" in results:
        formatted.append(f"Search Query: {results['query']}")
        
    return "\n".join(formatted)

# Tools
@mcp.tool()
async def web_search(query: str, count: int = 10, country: Optional[str] = None) -> str:
    """Search the web for information on a specific topic.
    
    Args:
        query: The search query
        count: Number of results (1-20, default 10)
        country: Optional country code for localized results (e.g., US, DE, JP)
    """
    if not BRAVE_API_KEY:
        return "Error: Brave Search API key is missing. Please set the BRAVE_API_KEY environment variable."
    
    try:
        results = await perform_web_search(
            query=query,
            count=count,
            country=country
        )
        
        return format_web_search_results(results)
    except Exception as e:
        logger.error(f"Error in web_search tool: {e}")
        return f"Error performing web search: {str(e)}"

@mcp.tool()
async def local_search(query: str, location: Optional[str] = None, count: int = 5) -> str:
    """Search for local businesses, places, and attractions.
    
    Args:
        query: The search query (e.g., "pizza restaurants", "hotels")
        location: Optional location specification (e.g., "New York, NY")
        count: Number of results (1-20, default 5)
    """
    if not BRAVE_API_KEY:
        return "Error: Brave Search API key is missing. Please set the BRAVE_API_KEY environment variable."
    
    try:
        # Combine location into query if provided
        search_query = query
        if location:
            search_query = f"{query} near {location}"
        
        results = await perform_local_search(
            query=search_query,
            count=count
        )
        
        return format_local_search_results(results)
    except Exception as e:
        logger.error(f"Error in local_search tool: {e}")
        return f"Error performing local search: {str(e)}"

@mcp.tool()
async def news_search(query: str, count: int = 10, country: Optional[str] = None, freshness: Optional[str] = None) -> str:
    """Search for news articles on a specific topic.
    
    Args:
        query: The search query
        count: Number of results (1-20, default 10)
        country: Optional country code for localized news (e.g., US, DE, JP)
        freshness: Optional freshness filter ("day", "week", "month")
    """
    if not BRAVE_API_KEY:
        return "Error: Brave Search API key is missing. Please set the BRAVE_API_KEY environment variable."
    
    try:
        results = await perform_news_search(
            query=query,
            count=count,
            country=country,
            freshness=freshness
        )
        
        return format_news_search_results(results)
    except Exception as e:
        logger.error(f"Error in news_search tool: {e}")
        return f"Error performing news search: {str(e)}"

@mcp.tool()
async def research_topic(query: str, include_news: bool = True) -> str:
    """Perform comprehensive research on a topic using both web and news search.
    
    Args:
        query: The research query
        include_news: Whether to include news results (default: True)
    """
    if not BRAVE_API_KEY:
        return "Error: Brave Search API key is missing. Please set the BRAVE_API_KEY environment variable."
    
    try:
        # Get web results
        web_results = await perform_web_search(query=query, count=5)
        
        # Get news results if requested
        news_results = None
        if include_news:
            news_results = await perform_news_search(query=query, count=3, freshness="week")
        
        # Format the combined results
        formatted = ["Research Results:"]
        formatted.append("\n## Web Information\n")
        
        if "web" in web_results and web_results["web"].get("results"):
            for idx, result in enumerate(web_results["web"]["results"], 1):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                description = result.get("description", "No description")
                
                formatted.append(f"{idx}. {title}")
                formatted.append(f"   URL: {url}")
                formatted.append(f"   {description}")
                formatted.append("")
        else:
            formatted.append("No web results found.")
        
        if include_news and news_results:
            formatted.append("\n## Recent News\n")
            
            if "news" in news_results and news_results["news"].get("results"):
                for idx, result in enumerate(news_results["news"]["results"], 1):
                    title = result.get("title", "No title")
                    source = result.get("source", "Unknown source")
                    published_time = result.get("published_time", "")
                    url = result.get("url", "No URL")
                    
                    formatted.append(f"{idx}. {title}")
                    formatted.append(f"   Source: {source}")
                    if published_time:
                        formatted.append(f"   Published: {published_time}")
                    formatted.append(f"   URL: {url}")
                    formatted.append("")
            else:
                formatted.append("No news results found.")
        
        formatted.append(f"Search Query: {query}")
        
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"Error in research_topic tool: {e}")
        return f"Error performing research: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
