import pandas as pd
import tensorflow as tf
import tf_keras as keras
# from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from transformers import BertTokenizerFast, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split


DATASET_FILE_PATH = "./data/processed/reddit_mental_health_clean.csv"


df = pd.read_csv(DATASET_FILE_PATH, index_col=0)
df = df.dropna(subset=["clean_text"])

print("Dataset shape:", df.shape)
print(df['target'].value_counts())

# tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Convert text to input IDs + attention masks
encodings = tokenizer(
    df["clean_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128,   # limit sequence length
    return_tensors="tf"
)


# X_train, X_val, y_train, y_val = train_test_split(
#     encodings["input_ids"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
# )

# X_train_mask, X_val_mask = train_test_split(
#     encodings["attention_mask"], test_size=0.2, random_state=42, stratify=df["target"]
# )


# train_dataset = tf.data.Dataset.from_tensor_slices((
#     {"input_ids": X_train, "attention_mask": X_train_mask},
#     y_train
# )).batch(16)

# val_dataset = tf.data.Dataset.from_tensor_slices((
#     {"input_ids": X_val, "attention_mask": X_val_mask},
#     y_val
# )).batch(16)

# Convert to numpy arrays before splitting
input_ids = encodings["input_ids"].numpy()
attention_mask = encodings["attention_mask"].numpy()
labels = df["target"].to_numpy()

# Split input_ids + labels
X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    input_ids, labels, test_size=0.2, random_state=42, stratify=labels
)

# Split attention_mask with same stratification
X_train_mask, X_val_mask, _, _ = train_test_split(
    attention_mask, labels, test_size=0.2, random_state=42, stratify=labels
)

# Build datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_train_ids, "attention_mask": X_train_mask},
    y_train
)).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_val_ids, "attention_mask": X_val_mask},
    y_val
)).batch(16)


# model = TFDistilBertForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased",
#     from_pt=False,
#     num_labels=df["target"].nunique()
# )
model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    from_pt=True,
    num_labels=df["target"].nunique()
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
# -------------------------
# Phase 1: Train classifier head (freeze base model)
# -------------------------
# for layer in model.bert.layers:
#     layer.trainable = False

# EXPERIMENT
# for layer in model.bert.layers[:8]:
#     layer.trainable = False
model.bert.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    # metrics=["accuracy"]
)

print("\nPhase 1: Training classifier head only...\n")
history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[early_stopping]
)

# -------------------------
# Phase 2: Fine-tune entire model
# -------------------------
# for layer in model.bert.layers:
#     layer.trainable = True

# EXPERIMENT
# for layer in model.bert.layers[8:]:
#     layer.trainable = True
model.bert.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    # metrics=["accuracy"]
)

print("\nPhase 2: Fine-tuning full model...\n")
history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
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
model.save_pretrained("./models/bert-mental-health")
tokenizer.save_pretrained("./models/bert-mental-health")

# -------------------------
# Evaluation
# -------------------------
eval_results = model.evaluate(val_dataset)
print("\nFinal Evaluation on Validation Set:")
print(dict(zip(model.metrics_names, eval_results)))
