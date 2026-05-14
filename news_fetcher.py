import feedparser
import pandas as pd
from datetime import datetime
import hashlib

# Free RSS feeds — no API key needed
RSS_FEEDS = {
    "Technology": [
        "https://feeds.feedburner.com/TechCrunch",
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
    ],
    "Business": [
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    ],
    "Science": [
        "https://www.sciencedaily.com/rss/all.xml",
        "https://feeds.feedburner.com/NationalGeographic",
    ],
    "World": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    ],
    "Health": [
        "https://feeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
        "https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml",
    ],
}

# Fallback sample data when feeds are unavailable (e.g. no internet on Streamlit)
FALLBACK_DATA = [
    {
        "title": "OpenAI releases new reasoning model with improved coding abilities",
        "summary": "OpenAI has announced a major update to its flagship model, featuring significantly improved performance on coding benchmarks and mathematical reasoning tasks.",
        "source": "TechCrunch",
        "category": "Technology",
        "url": "https://techcrunch.com",
        "published": "2025-05-15",
    },
    {
        "title": "Global markets rise as inflation data comes in lower than expected",
        "summary": "Stock markets worldwide posted gains after new data showed consumer price inflation cooling faster than analysts had predicted, raising hopes for interest rate cuts.",
        "source": "CNBC",
        "category": "Business",
        "url": "https://cnbc.com",
        "published": "2025-05-15",
    },
    {
        "title": "Scientists discover potential new treatment for Alzheimer's disease",
        "summary": "Researchers at Johns Hopkins University have identified a protein that may halt the progression of Alzheimer's, opening doors to a new class of treatments.",
        "source": "Science Daily",
        "category": "Health",
        "url": "https://sciencedaily.com",
        "published": "2025-05-14",
    },
    {
        "title": "NASA's Mars rover finds evidence of ancient water flow",
        "summary": "The Perseverance rover has captured images and chemical signatures strongly suggesting that liquid water once flowed across the Martian surface billions of years ago.",
        "source": "National Geographic",
        "category": "Science",
        "url": "https://nationalgeographic.com",
        "published": "2025-05-14",
    },
    {
        "title": "Tech giants face new antitrust regulations in the European Union",
        "summary": "The EU has proposed sweeping new regulations targeting large technology companies, requiring them to open their platforms to competitors and increase data transparency.",
        "source": "The Verge",
        "category": "Technology",
        "url": "https://theverge.com",
        "published": "2025-05-13",
    },
    {
        "title": "Renewable energy surpasses coal in global electricity generation",
        "summary": "For the first time in history, renewable energy sources including solar and wind have generated more electricity globally than coal-fired power plants.",
        "source": "BBC News",
        "category": "World",
        "url": "https://bbc.com",
        "published": "2025-05-13",
    },
    {
        "title": "New study links sleep quality to long-term cognitive health",
        "summary": "A large-scale study spanning 20 years has found strong correlations between sleep quality in middle age and cognitive decline risk later in life.",
        "source": "WebMD",
        "category": "Health",
        "url": "https://webmd.com",
        "published": "2025-05-12",
    },
    {
        "title": "Python overtakes JavaScript as most popular programming language",
        "summary": "The latest Stack Overflow developer survey shows Python has taken the top spot from JavaScript for the first time, driven by growth in data science and AI development.",
        "source": "Ars Technica",
        "category": "Technology",
        "url": "https://arstechnica.com",
        "published": "2025-05-12",
    },
    {
        "title": "Electric vehicle sales hit record high in Q1 2025",
        "summary": "Global electric vehicle sales reached an all-time high in the first quarter of 2025, with China, Europe, and the United States all reporting double-digit growth.",
        "source": "Bloomberg",
        "category": "Business",
        "url": "https://bloomberg.com",
        "published": "2025-05-11",
    },
    {
        "title": "WHO warns of rising antibiotic resistance worldwide",
        "summary": "The World Health Organization has issued an urgent warning about the growing threat of antibiotic-resistant bacteria, calling for immediate global action.",
        "source": "WHO",
        "category": "Health",
        "url": "https://who.int",
        "published": "2025-05-11",
    },
]


def _parse_date(entry):
    """Extract and format published date from RSS entry."""
    try:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d')
    except Exception:
        pass
    return datetime.today().strftime('%Y-%m-%d')


def _article_id(title, source):
    """Generate a unique ID for deduplication."""
    return hashlib.md5(f"{title}{source}".encode()).hexdigest()


def fetch_news(categories=None, max_per_feed=5):
    """
    Fetch news articles from RSS feeds.
    Falls back to sample data if feeds are unreachable.

    Args:
        categories: list of category names to fetch (None = all)
        max_per_feed: max articles per RSS feed

    Returns:
        pd.DataFrame with columns: title, summary, source, category, url, published
    """
    selected = categories or list(RSS_FEEDS.keys())
    articles = []
    seen_ids = set()

    for category in selected:
        if category not in RSS_FEEDS:
            continue
        for feed_url in RSS_FEEDS[category]:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_per_feed]:
                    title   = getattr(entry, 'title', '').strip()
                    summary = getattr(entry, 'summary', '') or getattr(entry, 'description', '')
                    # Strip HTML tags from summary
                    import re
                    summary = re.sub(r'<[^>]+>', '', summary).strip()
                    summary = summary[:500] if summary else title

                    source  = feed.feed.get('title', feed_url.split('/')[2])
                    url     = getattr(entry, 'link', feed_url)
                    pub     = _parse_date(entry)

                    uid = _article_id(title, source)
                    if uid in seen_ids or not title:
                        continue
                    seen_ids.add(uid)

                    articles.append({
                        "title": title,
                        "summary": summary,
                        "source": source,
                        "category": category,
                        "url": url,
                        "published": pub,
                    })
            except Exception:
                continue  # silently skip broken feeds

    if not articles:
        # Filter fallback data by selected categories
        articles = [a for a in FALLBACK_DATA if a["category"] in selected]

    df = pd.DataFrame(articles)
    df['published'] = pd.to_datetime(df['published'])
    df = df.sort_values('published', ascending=False).reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = fetch_news()
    print(f"Fetched {len(df)} articles")
    print(df[['title', 'category', 'source']].head(10))
