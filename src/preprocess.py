"""
NLP Preprocessing Pipeline
Handles text cleaning, tokenization, stopword removal, and stemming.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove URLs, HTML tags, punctuation, digits
    3. Remove stopwords
    4. Stem tokens
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize and filter stopwords, then stem
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]

    return " ".join(tokens)
