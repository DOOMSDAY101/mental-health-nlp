---
# 🧠 Mental Health Text Analysis using NLP

This project explores **Natural Language Processing (NLP)** for detecting signs of **mental health issues** in Reddit posts.
It applies **deep learning with Transformers (DistilBERT)** to classify posts into categories such as **Stress, Depression, Bipolar disorder, Personality disorder, and Anxiety**.
---

## 📌 Project Overview

- **Goal**: Analyze social media/text data to detect mental health concerns.
- **Dataset**: Reddit posts sourced from mental health–related subreddits.
- **Tasks**:

  - Preprocess and clean raw text (titles + content).
  - Augment data using paraphrasing (T5-based).
  - Train a **DistilBERT classifier** for multi-class classification.
  - Evaluate performance with accuracy, loss, classification report, and confusion matrix.
  - Deploy an interactive **predictor script** for inference.

- **Frameworks**: TensorFlow, HuggingFace Transformers, NLTK.
- **SDG Link**: Contributes to **SDG 3 – Good Health and Well-being**.
- **Ethics**: Mental health data is sensitive — privacy, anonymity, and ethical handling are emphasized.

---

## 📂 Project Structure

```
mental-health-nlp/
│
├── data/
│   ├── raw/                # Original dataset (Reddit posts)
│   └── processed/          # Cleaned + augmented dataset
│
├── models/
│   └── bert-mental-health/ # Saved fine-tuned DistilBERT model
│
├── src/
│   ├── preprocessing.py    # Data cleaning & augmentation
│   ├── train.py            # Model training & fine-tuning
│   ├── evaluate.py         # Model evaluation & confusion matrix
│   └── predict.py          # Inference / interactive prediction
│
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

---

## 📊 Dataset

Each Reddit post contains:

- **`text`** → Full content (main input for NLP tasks).
- **`title`** → Post title (optional contextual feature).
- **`target`** → Label for classification.

### Label Mapping

```
0 = Stress
1 = Depression
2 = Bipolar disorder
3 = Personality disorder
4 = Anxiety
```

---

## 🔄 Preprocessing

Implemented in [`preprocessing.py`](src/preprocessing.py):

- **Text cleaning**:

  - Lowercasing
  - Remove URLs, mentions, markdown links
  - Remove emojis, punctuation, numbers
  - Normalize whitespace

- **Optional Augmentation** (using `Vamsi/T5_Paraphrase_Paws`):

  - Generates paraphrased variants of posts to improve model generalization.
  - Saves augmented dataset for training.

---

## 🏗️ Model Training

Training pipeline in [`train.py`](src/train.py):

1. Load and preprocess dataset (`clean_text`).
2. Tokenize using `DistilBertTokenizerFast`.
3. Train/validation split (80/20).
4. **Two-phase training**:

   - **Phase 1**: Freeze all but last 2 transformer layers, train classifier head.
   - **Phase 2**: Unfreeze full model, fine-tune with class weights.

5. Callbacks:

   - Early stopping
   - Checkpoint (save best model)

6. Save fine-tuned model + tokenizer.

### Final Evaluation Example

```
Final Evaluation on Validation Set:
{'loss': 0.0748, 'accuracy': 0.9811}
```

---

## 📈 Model Evaluation

Implemented in [`evaluate.py`](src/evaluate.py):

- Reloads trained model + tokenizer.
- Evaluates on validation set with **loss & accuracy**.
- Generates:

  - **Classification Report** (precision, recall, f1-score per class).
  - **Confusion Matrix** (visualized with Matplotlib).

- Progress bars (via `tqdm`) show prediction progress.

---

## 🤖 Prediction / Inference

Interactive script [`predict.py`](src/predict.py):

```bash
$ python src/predict.py
Mental Health Text Classifier (type 'x' to exit)

Enter a text to classify: I've been feeling very anxious lately, can't sleep.
{'text': "I've been feeling very anxious lately, can't sleep.",
 'clean_text': 'ive been feeling very anxious lately cant sleep',
 'predicted_class': 'Anxiety',
 'confidence': 0.945}
--------------------------------------------------
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/mental-health-nlp.git
cd mental-health-nlp
```

### Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies**:

- `tensorflow` / `tf-keras`
- `transformers`
- `torch` (for paraphrasing model)
- `nltk`
- `emoji`
- `scikit-learn`
- `matplotlib` / `seaborn`

---

## 🔒 Ethics & Privacy

⚠️ **Important**:

- Reddit posts contain **sensitive mental health data**.
- Data must be anonymized and handled responsibly.
- Models are **research tools only** — not substitutes for clinical diagnosis or therapy.

---

## 🌍 SDG Contribution

This project aligns with **United Nations Sustainable Development Goal 3 (SDG 3): Good Health and Well-being**, by exploring how AI can help in **early detection of mental health issues** through text analysis.

---

## 🚀 Next Steps

- Deploy as a **web app** (Streamlit / Flask).
- Explore **multi-modal models** (title + text + metadata).
- Experiment with **Topic Modelling** (e.g., BERTopic, LDA) for unsupervised insights.
- Extend dataset with other sources (forums, blogs).

---

## 👨‍💻 Author

Developed by **Ifeoluwa Sulaiman**
For **Mental Health Text Analysis using NLP**

---
