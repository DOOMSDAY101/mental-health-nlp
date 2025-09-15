import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


from preprocessing import clean_text  

# -------------------------
# Config
# -------------------------
MODEL_PATH = "./models/bert-mental-health"
DATASET_FILE_PATH = "./data/processed/reddit_mental_health_clean.csv"

LABEL_MAP = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar disorder",
    3: "Personality disorder",
    4: "Anxiety"
}

BATCH_SIZE = 32 

# -------------------------
# Load model & tokenizer
# -------------------------
print("Loading model...")
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)


# -------------------------
# Load dataset
# -------------------------
print("Loading dataset...")
df = pd.read_csv(DATASET_FILE_PATH, index_col=0)
df = df.dropna(subset=["clean_text"])

# Clean text again just in case
df["clean_text"] = df["clean_text"].apply(clean_text)

X_texts = df["clean_text"].tolist()
y_true = df["target"].to_numpy()

# -------------------------
# Tokenize
# -------------------------
encodings = tokenizer(
    X_texts,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="tf"
)

dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}
)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]},
    y_true
)).batch(BATCH_SIZE)

# Dataset for predictions (no labels needed)
pred_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}
)).batch(BATCH_SIZE)

# -------------------------
# Evaluation
# -------------------------
print("\nEvaluating model on validation set...")
eval_results = model.evaluate(val_dataset)
print(dict(zip(model.metrics_names, eval_results)))

# -------------------------
# Run predictions
# -------------------------
print("\nRunning predictions...")
y_pred = []

for batch in tqdm(pred_dataset, desc="Predicting", unit="batch"):
    outputs = model(batch, training=False)
    logits = outputs.logits
    preds = tf.argmax(logits, axis=-1).numpy()
    y_pred.extend(preds)

y_pred = np.array(y_pred)

# -------------------------
# Classification report
# -------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values())))

# -------------------------
# Confusion matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=LABEL_MAP.values(),
    yticklabels=LABEL_MAP.values(),
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
