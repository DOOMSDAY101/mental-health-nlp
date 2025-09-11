# from transformers import BertTokenizerFast, TFBertForSequenceClassificationimport
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
from utils import clean_text


# Load model & tokenizer
model_path = "./models/bert-mental-health"
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

LABEL_MAP = {0: "Stress", 1: "Depression", 2: "Bipolar disorder", 3: "Personality disorder", 4: "Anxiety"}  


def predict_text(text: str):
    # Step 1: clean
    cleaned = clean_text(text)

    # Step 2: tokenize (match training setup)
    encodings = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

    # Step 3: run model
    outputs = model(encodings)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)
    predicted_class = int(tf.argmax(probs, axis=-1).numpy()[0])

    return {
        "text": text,
        "clean_text": cleaned,
        "predicted_class": LABEL_MAP[predicted_class],
        "confidence": float(tf.reduce_max(probs).numpy())
    }


if __name__ == "__main__":
    sample_text = "I've been feeling really down lately and can't focus."
    user_text = input("Enter a text to classify: ")
    result = predict_text(user_text)
    print(result)
