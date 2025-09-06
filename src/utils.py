# predict.py

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords + lemmatizer (same as training)
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Basic text cleaning (same as training)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "<URL>", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "<URL>", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)
