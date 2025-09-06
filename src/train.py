import pandas as pd
import tensorflow as tf
import keras
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split


DATASET_FILE_PATH = "./data/processed/reddit_mental_health_clean.csv"


df = pd.read_csv(DATASET_FILE_PATH, index_col=0)
df = df.dropna(subset=["clean_text"])

print("Dataset shape:", df.shape)
print(df['target'].value_counts())

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Convert text to input IDs + attention masks
encodings = tokenizer(
    df["clean_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128,   # limit sequence length
    return_tensors="tf"
)


X_train, X_val, y_train, y_val = train_test_split(
    encodings["input_ids"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
)

X_train_mask, X_val_mask = train_test_split(
    encodings["attention_mask"], test_size=0.2, random_state=42, stratify=df["target"]
)


train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_train, "attention_mask": X_train_mask},
    y_train
)).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_val, "attention_mask": X_val_mask},
    y_val
)).batch(16)

model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=df["target"].nunique()
)

# -------------------------
# Phase 1: Train classifier head (freeze base model)
# -------------------------
for layer in model.distilbert.layers:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

print("\nPhase 1: Training classifier head only...\n")
history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2
)

# -------------------------
# Phase 2: Fine-tune entire model
# -------------------------
for layer in model.distilbert.layers:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

print("\nPhase 2: Fine-tuning full model...\n")
history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

print("\nModel Summary:")
model.summary()


# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# metrics = ["accuracy"]

# model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=3
# )


# -------------------------
# Save model & tokenizer
# -------------------------
model.save_pretrained("./models/distilbert-mental-health")
tokenizer.save_pretrained("./models/distilbert-mental-health")

# -------------------------
# Evaluation
# -------------------------
eval_results = model.evaluate(val_dataset)
print("\nFinal Evaluation on Validation Set:")
print(dict(zip(model.metrics_names, eval_results)))
