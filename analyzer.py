import re
import string
import pandas as pd
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)


# ── Sentiment Model ───────────────────────────────────────────────────────────

def _build_sentiment_model():
    """Train a Logistic Regression sentiment classifier."""
    texts = [
        # Positive
        "breakthrough discovery amazing innovation success achieved excellent results",
        "markets surge record high growth profit gains investors celebrate",
        "scientists find cure treatment hope recovery health improvement",
        "new technology revolutionizes industry boosts productivity efficiency",
        "economy grows jobs created unemployment falls prosperity increases",
        "award winning performance outstanding achievement celebrated worldwide",
        "peace agreement signed cooperation partnership flourishes progress",
        "renewable energy clean future sustainable environment protected",

        # Negative
        "crisis disaster collapse failure loss damage destruction catastrophe",
        "markets crash recession unemployment rises poverty concern alarm",
        "outbreak disease epidemic threat warning dangerous deadly spread",
        "attack violence conflict war deaths casualties tragedy grief",
        "scandal corruption fraud arrested charged guilty convicted crime",
        "pollution climate emergency threat extinction endangered warning",
        "protest unrest anger frustration crisis government fails people",
        "company bankrupt layoffs workers fired job losses economic pain",

        # Neutral
        "officials announced new policy review committee meeting scheduled",
        "report published data shows statistics figures analysis released",
        "company announced quarterly results expected targets revenue",
        "government released statement regarding ongoing discussions talks",
        "research study conducted university published journal findings",
        "election results counted votes tallied officials confirm outcome",
        "market closed trading volume average standard performance noted",
        "weather forecast temperature expected rainfall region area",
    ]
    labels = ['positive'] * 8 + ['negative'] * 8 + ['neutral'] * 8
    processed = [preprocess(t) for t in texts]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(processed, labels)
    return pipeline


_model = None

def get_sentiment_model():
    global _model
    if _model is None:
        _model = _build_sentiment_model()
    return _model


def analyze_sentiment(text: str) -> dict:
    """
    Predict sentiment label and confidence scores for text.

    Returns:
        dict with 'label' and 'scores' keys
    """
    model   = get_sentiment_model()
    cleaned = preprocess(text)
    label   = model.predict([cleaned])[0]
    proba   = model.predict_proba([cleaned])[0]
    classes = model.classes_
    scores  = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
    return {'label': label, 'scores': scores}


# ── Keyword Extraction ────────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top keywords from text using TF-IDF on single document."""
    cleaned = preprocess(text)
    words   = cleaned.split()
    # Frequency-based fallback for short texts
    if len(words) < 5:
        return words[:top_n]

    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_n)]


# ── Batch Analysis ────────────────────────────────────────────────────────────

def analyze_dataframe(df: pd.DataFrame, text_col: str = "summary") -> pd.DataFrame:
    """
    Add sentiment and keyword columns to a DataFrame.

    Args:
        df:       Input DataFrame
        text_col: Column with text to analyze

    Returns:
        DataFrame with added columns: sentiment, sentiment_scores, keywords
    """
    df = df.copy()
    results  = df[text_col].apply(lambda t: analyze_sentiment(str(t)))
    df['sentiment']        = results.apply(lambda r: r['label'])
    df['sentiment_scores'] = results.apply(lambda r: r['scores'])
    df['keywords']         = df[text_col].apply(lambda t: extract_keywords(str(t)))
    return df


# ── Category Distribution ─────────────────────────────────────────────────────

def category_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize sentiment distribution per news category.

    Returns:
        DataFrame with category and sentiment counts
    """
    return (
        df.groupby(['category', 'sentiment'])
          .size()
          .reset_index(name='count')
    )


if __name__ == '__main__':
    sample_texts = [
        "Scientists make breakthrough discovery that could cure cancer.",
        "Stock market crashes as recession fears grip investors.",
        "Government officials met today to discuss new trade policy.",
    ]
    for text in sample_texts:
        result = analyze_sentiment(text)
        kw     = extract_keywords(text)
        print(f"\nText      : {text}")
        print(f"Sentiment : {result['label'].upper()}")
        print(f"Scores    : {result['scores']}")
        print(f"Keywords  : {kw}")
