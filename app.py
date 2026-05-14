import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from news_fetcher import fetch_news, RSS_FEEDS
from summarizer import summarize_dataframe
from analyzer import analyze_dataframe, category_sentiment_summary

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsLens — AI News Dashboard",
    page_icon="📰",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1 { font-family: 'Playfair Display', serif; font-size: 2.8rem !important; letter-spacing: -1px; }
h2, h3 { font-family: 'Playfair Display', serif; }

.metric-card {
    background: #0f172a;
    color: white;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    border: 1px solid #1e293b;
}
.metric-card .number { font-size: 2.2rem; font-weight: 700; color: #38bdf8; }
.metric-card .label  { font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.news-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    border-left: 4px solid #38bdf8;
    transition: box-shadow 0.2s;
}
.news-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
.news-card .headline { font-size: 1rem; font-weight: 600; color: #0f172a; margin-bottom: 6px; line-height: 1.4; }
.news-card .summary  { font-size: 0.88rem; color: #475569; line-height: 1.6; margin-bottom: 10px; }
.news-card .meta     { font-size: 0.78rem; color: #94a3b8; }

.badge-positive { background: #dcfce7; color: #166534; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-negative { background: #fee2e2; color: #991b1b; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-neutral  { background: #fef9c3; color: #854d0e; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

.keyword-tag {
    display: inline-block;
    background: #f1f5f9;
    color: #334155;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    margin: 2px;
}

.stButton > button {
    background: #0f172a;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
}
.stButton > button:hover { background: #1e293b; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📰 NewsLens")
    st.markdown("*AI-powered news aggregator*")
    st.markdown("---")

    all_categories = list(RSS_FEEDS.keys())
    selected_cats  = st.multiselect(
        "Filter by Category",
        options=all_categories,
        default=all_categories,
    )

    sentiment_filter = st.selectbox(
        "Filter by Sentiment",
        options=["All", "Positive", "Negative", "Neutral"],
    )

    summarize_toggle = st.toggle("Show AI Summaries", value=True)
    max_articles     = st.slider("Max Articles to Show", 5, 30, 12)

    st.markdown("---")
    refresh = st.button("🔄 Refresh News")

    st.markdown("---")
    st.markdown("<small>Built with Python · RSS Feeds · NLP · Plotly · Streamlit</small>", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_data(categories):
    df = fetch_news(categories=categories, max_per_feed=6)
    df = analyze_dataframe(df, text_col="summary")
    df = summarize_dataframe(df, text_col="summary", sentences=2)
    return df

if refresh:
    st.cache_data.clear()

with st.spinner("Fetching and analyzing news..."):
    df = load_data(tuple(selected_cats) if selected_cats else tuple(all_categories))


# ── Apply Filters ─────────────────────────────────────────────────────────────
filtered_df = df.copy()
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter.lower()]
filtered_df = filtered_df.head(max_articles)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📰 NewsLens")
st.markdown(f"*AI-analyzed news dashboard · Last updated: {datetime.now().strftime('%B %d, %Y %H:%M')}*")
st.markdown("---")


# ── Metrics Row ───────────────────────────────────────────────────────────────
total     = len(df)
positive  = len(df[df['sentiment'] == 'positive'])
negative  = len(df[df['sentiment'] == 'negative'])
neutral   = len(df[df['sentiment'] == 'neutral'])
pos_pct   = round(positive / total * 100) if total else 0

col1, col2, col3, col4, col5 = st.columns(5)
for col, number, label in [
    (col1, total,    "Total Articles"),
    (col2, len(selected_cats or all_categories), "Categories"),
    (col3, positive, "😊 Positive"),
    (col4, negative, "😞 Negative"),
    (col5, neutral,  "😐 Neutral"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="number">{number}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Charts Row ────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    color_map = {'positive': '#22c55e', 'negative': '#ef4444', 'neutral': '#eab308'}
    fig_pie = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Count',
        color='Sentiment',
        color_discrete_map=color_map,
        hole=0.45,
    )
    fig_pie.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    st.markdown("### Articles by Category")
    cat_summary = category_sentiment_summary(df)
    fig_bar = px.bar(
        cat_summary,
        x='category',
        y='count',
        color='sentiment',
        color_discrete_map=color_map,
        barmode='stack',
    )
    fig_bar.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="",
        yaxis_title="Articles",
        legend_title="Sentiment",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── News Cards ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 📋 Latest Articles ({len(filtered_df)} shown)")

if filtered_df.empty:
    st.info("No articles match the selected filters.")
else:
    for _, row in filtered_df.iterrows():
        sentiment = row['sentiment']
        badge_cls = f"badge-{sentiment}"
        emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
        emoji      = emoji_map.get(sentiment, '')

        summary_text = row['short_summary'] if summarize_toggle else row['summary']
        keywords_html = " ".join(
            f'<span class="keyword-tag">{kw}</span>'
            for kw in (row.get('keywords') or [])[:5]
        )

        pub_date = row['published'].strftime('%b %d, %Y') if pd.notna(row['published']) else ''

        st.markdown(f"""
        <div class="news-card">
            <div class="headline">
                <a href="{row['url']}" target="_blank" style="text-decoration:none; color:#0f172a;">
                    {row['title']}
                </a>
            </div>
            <div class="summary">{summary_text}</div>
            <div style="margin-bottom:8px">{keywords_html}</div>
            <div class="meta">
                <span class="{badge_cls}">{emoji} {sentiment.capitalize()}</span>
                &nbsp;·&nbsp; 📰 {row['source']}
                &nbsp;·&nbsp; 🏷 {row['category']}
                &nbsp;·&nbsp; 📅 {pub_date}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Raw Data ──────────────────────────────────────────────────────────────────
with st.expander("📊 View Raw Data Table"):
    display_cols = ['title', 'category', 'source', 'sentiment', 'published']
    st.dataframe(
        df[display_cols].rename(columns={c: c.title() for c in display_cols}),
        use_container_width=True,
        hide_index=True,
    )
    csv = df[display_cols].to_csv(index=False)
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name="news_analysis.csv",
        mime="text/csv",
    )
