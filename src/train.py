import pandas as pd
import tensorflow as tf
import tf_keras as keras
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, DistilBertConfig
# from transformers import BertTokenizerFast, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# from transformers import BertConfig
# model = TFBertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     config=config
# )


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


config = DistilBertConfig.from_pretrained("distilbert-base-uncased",
                                    num_labels=df["target"].nunique(),
                                    hidden_dropout_prob=0.4,
                                    attention_probs_dropout_prob=0.4) 

model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    # from_pt=True,
    # num_labels=df["target"].nunique()
    config=config
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
checkpoint = keras.callbacks.ModelCheckpoint(
    "./models/best_bert",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False
)

# -------------------------
# Phase 1: Train classifier head (freeze base model)
# -------------------------
# for layer in model.bert.layers:
#     layer.trainable = False

# EXPERIMENT

# for layer in model.bert.encoder.layer[:-4]:  # freeze all except last 4 layers
#     layer.trainable = False
for layer in model.distilbert.transformer.layer[:-2]:
    layer.trainable = False


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)

print("\nPhase 1: Training classifier head only...\n")
history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
)

# -------------------------
# Phase 2: Fine-tune entire model
# -------------------------
# for layer in model.bert.layers:
#     layer.trainable = True

# EXPERIMENT
model.distilbert.trainable = True
# Unfreeze last 4 layers
# for layer in model.bert.encoder.layer[-4:]:
#     layer.trainable = True
# model.bert.trainable = True


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    # metrics=["accuracy"]
)


print("\nModel Summary:")
model.summary()



class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

print("\nPhase 2: Fine-tuning full model...\n")
history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint]
)


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
