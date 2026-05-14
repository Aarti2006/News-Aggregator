# 📰 NewsLens — AI-Powered News Aggregator & Analyzer

A real-time news dashboard that fetches live articles from RSS feeds, summarizes them using NLP, analyzes their sentiment, and visualizes insights — all built in Python with no API key required.

---

## 🚀 Live Demo

Deploy on Streamlit Cloud for free — see Setup below.

---

## ✨ Features

- 🌐 **Live News Fetching** — Pulls real articles from BBC, TechCrunch, The Verge, CNBC, Science Daily & more via RSS feeds
- 🧠 **NLP Summarization** — Condenses long articles into 2-sentence summaries using LSA (Latent Semantic Analysis)
- 📊 **Sentiment Analysis** — Classifies each article as Positive, Negative, or Neutral using a TF-IDF + Logistic Regression model
- 🔑 **Keyword Extraction** — Identifies top keywords from each article automatically
- 📈 **Interactive Charts** — Sentiment distribution pie chart and category breakdown bar chart (Plotly)
- 🔍 **Filters** — Filter by category (Tech, Business, Health, Science, World) and sentiment
- ⬇️ **CSV Export** — Download analyzed data as a spreadsheet
- ⚡ **No API Key Needed** — Uses free public RSS feeds

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| feedparser | Parse RSS feeds from news sites |
| pandas | Data processing & structuring |
| scikit-learn | TF-IDF vectorization + Logistic Regression classifier |
| sumy | Extractive text summarization (LSA algorithm) |
| NLTK | Stopwords, tokenization, lemmatization |
| Plotly | Interactive data visualizations |
| Streamlit | Web dashboard UI |

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/news-aggregator.git
cd news-aggregator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🧠 How It Works

```
RSS Feeds → news_fetcher.py → Raw Articles (pandas DataFrame)
                                      ↓
                             analyzer.py → Sentiment + Keywords
                                      ↓
                            summarizer.py → Short NLP Summaries
                                      ↓
                              app.py → Streamlit Dashboard + Charts
```

1. **news_fetcher.py** — Parses multiple RSS feeds using `feedparser`, deduplicates articles, and returns a clean `pandas` DataFrame. Falls back to sample data if feeds are unavailable.

2. **summarizer.py** — Uses `sumy`'s LSA (Latent Semantic Analysis) algorithm to extract the most important sentences from each article.

3. **analyzer.py** — Preprocesses text (lowercasing, stopword removal, lemmatization), vectorizes with TF-IDF, and classifies sentiment with Logistic Regression. Also extracts top keywords using word frequency.

4. **app.py** — Streamlit dashboard with sidebar filters, metric cards, Plotly charts, and styled news cards with clickable headlines.

---

## 📁 Project Structure

```
news-aggregator/
├── app.py              # Streamlit dashboard (main entry point)
├── news_fetcher.py     # RSS feed parsing & data collection
├── summarizer.py       # NLP text summarization
├── analyzer.py         # Sentiment analysis & keyword extraction
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## 🙋 Author

**Aarti Negi**  
[GitHub](https://github.com/YOUR_USERNAME) ·
