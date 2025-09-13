import pandas as pd
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import pipeline,  AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# -------------------------
# Download NLTK resources
# -------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# -------------------------
# Load Paraphrasing Model (once, for augmentation)
# -------------------------
# paraphrase_pipe = pipeline(
#     "text2text-generation",
#     model="Vamsi/T5_Paraphrase_Paws",
#     tokenizer="Vamsi/T5_Paraphrase_Paws",
#     use_fast=False
# )
model_name = "Vamsi/T5_Paraphrase_Paws"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

paraphrase_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)


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
    """
    Minimal cleaning for BERT/DistilBERT.
    - Lowercase
    - Replace URLs and usernames with placeholders
    - Keep stopwords and emotional words (important for mental health context)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()  # lowercase
    
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) # Replace URLs with a placeholder instead of removing
    text = re.sub(r"@\w+", "", text)          # replace mentions
    text = re.sub(r"\[.*?\]\(.*?\)", "", text) # Remove markdown links but keep the URL placeholder
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()  # Collapse multiple spaces into one
    # tokens = text.split()
    # tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS]
    # return " ".join(tokens)
    return text


# def generate_paraphrases(text, num_return_sequences=2):
#     """Generate paraphrased versions of text."""
#     if not isinstance(text, str) or not text.strip():
#         return []
#     prompt = f"paraphrase: {text} </s>"
#     outputs = paraphrase_pipe(
#         prompt,
#         max_length=256,
#         num_return_sequences=num_return_sequences,
#         num_beams=5,
#         temperature=1.5
#     )
#     return list({o["generated_text"].strip() for o in outputs})

def generate_paraphrases(text, num_return_sequences=2, max_retries=3):
    """Generate at least `num_return_sequences` unique paraphrases with retries."""
    if not isinstance(text, str) or not text.strip():
        return []

    prompt = f"paraphrase: {text}"
    unique_outputs = set()
    retries = 0

    while len(unique_outputs) < num_return_sequences and retries < max_retries:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # outputs = paraphrase_pipe(
        #     prompt,
        #     max_new_tokens=256,
        #     num_return_sequences=num_return_sequences,
        #     num_beams=5,
        # )
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=5,
            num_return_sequences=num_return_sequences,
            do_sample=False  # deterministic beam search
        )
        for o in outputs:
            paraphrase = tokenizer.decode(o, skip_special_tokens=True).strip()
            # unique_outputs.add(o["generated_text"].strip())
            unique_outputs.add(paraphrase)
        retries += 1

    # Ensure we only return exactly num_return_sequences
    return list(unique_outputs)[:num_return_sequences]



# def preprocess_dataframe(df, text_column="text"):
#     """Apply cleaning to the whole dataframe and create 'clean_text' column."""
#     df["clean_text"] = df[text_column].apply(clean_text)
#     return df
def preprocess_dataframe(df, text_column="text", augment=True):
    """Apply cleaning and optionally augment with paraphrases."""
    df["clean_text"] = df[text_column].apply(clean_text)

    if augment:
        augmented_rows = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating paraphrases"):
            paraphrases = generate_paraphrases(row["clean_text"])
            for p in paraphrases:
                augmented_rows.append({
                    "text": row["text"],
                    "clean_text": p,
                    "target": row.get("target", None)  # keep label if available
                })
        if augmented_rows:
            aug_df = pd.DataFrame(augmented_rows)
            df = pd.concat([df, aug_df], ignore_index=True)
            print(f"Augmented dataset with {len(aug_df)} paraphrased rows.")
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

    sample_text = df["text"].iloc[0]
    print("RAW:", sample_text)
    print("CLEANED:", clean_text(sample_text))
    print("PARAPHRASES:", generate_paraphrases(clean_text(sample_text)))

    # Preprocess text and save
    df = preprocess_dataframe(df)
    save_processed_data(df, processed_file)
