"""
Shared tech news tools for NLIP agent frameworks demo.
These tools integrate with an external news API to fetch recent articles.
"""

import os
import httpx
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"


async def google_search_news(query: str, num: int = 10) -> list[dict]:
    if not SERPER_API_KEY:
        return []

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num}

    async with httpx.AsyncClient() as client:
        r = await client.post(SERPER_URL, headers=headers, json=payload, timeout=20.0)
        r.raise_for_status()
        data = r.json()

    return [
        {"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")}
        for item in data.get("organic", [])
    ]


async def get_tech_news_brief(
    topic: str,
    days: int = 1,
    domains: str | None = None,
    source: str = "serper",  # "serper" or "newsapi"
) -> str:
    """
    Fetch recent tech news about a topic using either:
    - Serper (Google-style search), or
    - NewsAPI (optional fallback).
    """

    if source == "serper":
        query = f"{topic} technology OR AI OR cybersecurity news last {days} days"
        results = await google_search_news(query, num=10)
        if not results:
            return f"❌ No Google results for '{topic}' (or SERPER_API_KEY missing)."

        return "\n---\n".join(
            f"🔎 **{r.get('title','(no title)')}**\n"
            f"- Snippet: {r.get('snippet','(no snippet)')}\n"
            f"- URL: {r.get('link','(no link)')}"
            for r in results
        )

    # NewsAPI fallback
    if not NEWS_API_KEY:
        return "❌ NEWS_API_KEY is missing. Add it to your .env file or use source='serper'."

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    q = (
        f'({topic}) AND (technology OR tech OR software OR AI OR "artificial intelligence" '
        f'OR cybersecurity OR "information security" OR cloud OR "data center" OR semiconductors '
        f'OR GPU OR chip OR "export controls")'
    )

    params = {
        "q": q,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "searchIn": "title,description",
        "apiKey": NEWS_API_KEY,
        "pageSize": 20,
    }
    if domains:
        params["domains"] = domains

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(NEWS_API_URL, params=params, timeout=20.0)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return f"❌ Error fetching news: {str(e)}"

    articles = data.get("articles", [])
    if not articles:
        return f"❌ No news found about '{topic}' in the last {days} days."

    summaries = []
    for a in articles:
        summaries.append(
            f"📰 **{a.get('title','(no title)')}**\n"
            f"   - Source: {a.get('source',{}).get('name','(unknown)')}\n"
            f"   - Date: {a.get('publishedAt','(unknown)')}\n"
            f"   - Summary: {a.get('description','(No description)')}\n"
            f"   - URL: {a.get('url','(no url)')}\n"
        )

    return "\n---\n".join(summaries)
