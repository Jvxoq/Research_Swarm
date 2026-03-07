# app/core/langgraph/tools.py
"""Tools for research agents."""
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from app.core.config import settings
from app.core.logging import logger
import requests
from bs4 import BeautifulSoup


# Web search tool
def get_web_search_tool():
    """
    Get Tavily search tool.
    
    Returns 5 search results with title, URL, and snippet.
    """
    return TavilySearchResults(
        api_key=settings.tavily_api_key,
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )

# Url Fetch tool
@tool
def fetch_url(url: str) -> str:
    """
    Fetch and extract text content from a URL.
    
    Args:
        url: The URL to fetch
    
    Returns:
        Cleaned text content from the page
    """
    try:
        logger.info("fetching_url", url=url)
        
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Truncate to avoid token limits
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        logger.info("url_fetched", url=url, length=len(text))
        return text
        
    except Exception as e:
        logger.error("url_fetch_failed", url=url, error=str(e))
        return f"Error fetching URL: {str(e)}"