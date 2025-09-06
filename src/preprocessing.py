import pandas as pd
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# -------------------------
# Download NLTK resources
# -------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# -------------------------
# Functions
# -------------------------

def load_dataset(file_path):
    """
    Load CSV dataset into a Pandas DataFrame.
    Automatically uses first column as index if needed.
    """
    df = pd.read_csv(file_path, index_col=0)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_dataset(df, text_column="text"):
    """
    Drop rows with missing text and return cleaned dataframe.
    """
    initial_rows = df.shape[0]
    df = df.dropna(subset=[text_column])
    dropped_rows = initial_rows - df.shape[0]
    print(f"Dropped {dropped_rows} rows with missing '{text_column}'")
    return df

def inspect_dataset(df, num_rows=5, show_full_text=False):
    """
    Inspect dataset: columns, missing values, head.
    Can also display full text without truncation.
    """
    print("Columns:", df.columns.tolist())
    print("Missing values per column:\n", df.isnull().sum())

    if show_full_text:
        pd.set_option('display.max_colwidth', None)  # show full text

    print("\nSample rows:")
    print(df.head(num_rows))

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # lowercase
    # Replace URLs with a placeholder instead of removing
    text = re.sub(r"http\S+|www\S+|https\S+", "<URL>", text)
    # Remove markdown links but keep the URL placeholder
    text = re.sub(r"\[.*?\]\(.*?\)", "<URL>", text)
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation and numbers
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def preprocess_dataframe(df, text_column="text"):
    """Apply cleaning to the whole dataframe and create 'clean_text' column."""
    df["clean_text"] = df[text_column].apply(clean_text)
    return df


def save_processed_data(df, save_path):
    """Save cleaned dataframe to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ensure directory exists
    df.to_csv(save_path, index=False)
    print(f"Processed dataset saved to {save_path}")


def display_row_text(df, row_index):
    """
    Display full text for a specific row.
    """
    pd.set_option('display.max_colwidth', None)
    print(df.loc[row_index, 'text'])



if __name__ == "__main__":
    raw_file = os.path.join("data", "raw", "reddit_mental_health_nlp.csv")
    processed_file = os.path.join("data", "processed", "reddit_mental_health_clean.csv")


    # Load and inspect
    df = load_dataset(raw_file)
    df = clean_dataset(df)
    inspect_dataset(df, num_rows=5, show_full_text=True)

    # Preprocess text and save
    df = preprocess_dataframe(df)
    save_processed_data(df, processed_file)