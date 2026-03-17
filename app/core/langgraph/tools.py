from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from app.core.config import settings
from app.core.logging import logger
import requests
from bs4 import BeautifulSoup


# Web search tool
@tool
def web_search_tool(input: str):
    """
    Get Tavily search tool.

    Args:
        input: Search Query

    Returns:
        Dict with results, answer, etc.
    """
    tool = TavilySearch(
        search_depth="basic",
    )
    return tool.invoke(input)


# Url Fetch tool
@tool
def fetch_content(url: str) -> str:
    """
    Fetch and extract text content from a URL.

    Args:
        url: The URL to fetch

    Returns:
        Cleaned text content from the page
    """
    try:
        logger.info("fetching_url", url=url)

        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        # Truncate to avoid token limits
        max_chars = 5000  # 350 words
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        logger.info("url_fetched", url=url, length=len(text))
        return text

    except Exception as e:
        logger.error("url_fetch_failed", url=url, error=str(e))
        return f"Error fetching URL: {str(e)}"
