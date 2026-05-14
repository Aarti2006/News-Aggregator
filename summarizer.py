from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

LANGUAGE = "english"


def _get_summarizer(method="lsa"):
    """Return configured summarizer instance."""
    stemmer = Stemmer(LANGUAGE)
    if method == "lexrank":
        summarizer = LexRankSummarizer(stemmer)
    else:
        summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    return summarizer


def summarize(text: str, sentences: int = 2, method: str = "lsa") -> str:
    """
    Summarize text using extractive NLP summarization.

    Args:
        text:      Input text to summarize
        sentences: Number of sentences in summary
        method:    'lsa' (Latent Semantic Analysis) or 'lexrank'

    Returns:
        Summarized string
    """
    text = text.strip()
    if not text or len(text.split()) < 30:
        # Text too short to summarize — return as-is (truncated)
        return text[:300] + ("..." if len(text) > 300 else "")

    try:
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
        summarizer = _get_summarizer(method)
        summary_sentences = summarizer(parser.document, sentences)
        result = " ".join(str(s) for s in summary_sentences)
        return result if result.strip() else text[:300]
    except Exception:
        return text[:300] + ("..." if len(text) > 300 else "")


def summarize_dataframe(df, text_col="summary", sentences=2):
    """
    Add a 'short_summary' column to a DataFrame by summarizing text_col.

    Args:
        df:        pandas DataFrame
        text_col:  column containing text to summarize
        sentences: number of sentences per summary

    Returns:
        DataFrame with new 'short_summary' column
    """
    df = df.copy()
    df['short_summary'] = df[text_col].apply(
        lambda t: summarize(str(t), sentences=sentences)
    )
    return df


if __name__ == '__main__':
    sample = """
    Artificial intelligence has transformed the technology industry over the past decade.
    Major companies like Google, Microsoft, and Amazon have invested billions of dollars
    into AI research and development. These investments have led to breakthroughs in natural
    language processing, computer vision, and autonomous systems. Experts predict that AI
    will continue to reshape industries from healthcare to finance in the coming years.
    However, concerns about job displacement and ethical implications remain significant
    challenges that governments and organizations must address.
    """
    print("Original:", sample.strip())
    print("\nSummary:", summarize(sample, sentences=2))
