import requests
import pandas as pd
from config import NEWS_API_KEY

def fetch_news(query="supply chain disruption", page_size=20):
    """
    Fetch news articles from NewsAPI
    """

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error fetching news:", e)
        return pd.DataFrame()

    articles = response.json().get("articles", [])

    news_data = []

    for article in articles:
        news_data.append({
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "source": article["source"]["name"],
            "published_at": article.get("publishedAt"),
            "url": article.get("url")
        })

    df = pd.DataFrame(news_data)

    return df