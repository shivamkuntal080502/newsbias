import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ----------------------
# Helper functions
# ----------------------

def preprocess_df(df: pd.DataFrame):
    df = df[['sentence', 'type']].dropna()
    df['type'] = df['type'].str.lower().str.strip()
    df = df[df['type'].isin(['left', 'center', 'right'])]
    label_map = {'left': 0, 'center': 1, 'right': 2}
    df['label'] = df['type'].map(label_map)
    return df

def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer):
    return ds.map(lambda ex: tokenizer(
                       ex['sentence'],
                       padding='max_length',
                       truncation=True,
                       max_length=512
                   ), batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': cm}

@st.cache_resource
def load_trained(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left','Center','Right'],
                yticklabels=['Left','Center','Right'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# ----------------------
# Streamlit Layout
# ----------------------

st.set_page_config(page_title="Political Leaning Classifier", layout="wide")
st.title("📰 Political Leaning Classifier & Trainer")

mode = st.sidebar.radio("Choose Mode", ["Train & Evaluate", "Predict Only"])

if mode == "Train & Evaluate":
    st.header("1. Upload & Preprocess Data")
    uploaded = st.file_uploader("Upload labeled_dataset.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Raw data: {df.shape[0]} rows")
        df = preprocess_df(df)
        st.write(f"Filtered to 3 classes: {df.shape[0]} rows")
        st.write(df.head())

        st.header("2. Training Hyperparameters")
        epochs = st.number_input("Num Training Epochs", value=5, min_value=1)
        lr = st.number_input("Learning Rate", value=2e-5, format="%.1e")
        batch_size = st.selectbox("Per-Device Batch Size", [4, 8, 16], index=1)
        output_dir = st.text_input("Output Directory", value="./results")

        if st.button("Start Training"):
            with st.spinner("Tokenizing and splitting..."):
                train_df, val_df = train_test_split(df, test_size=0.2,
                                                    stratify=df['label'], random_state=42)
                train_ds = Dataset.from_pandas(train_df[['sentence','label']])
                val_ds = Dataset.from_pandas(val_df[['sentence','label']])

                tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                train_ds = tokenize_dataset(train_ds, tokenizer)
                val_ds = tokenize_dataset(val_ds, tokenizer)
                train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
                val_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

            with st.spinner("Initializing model and trainer..."):
                model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    learning_rate=lr,
                    weight_decay=0.01,
                    evaluation_strategy="epoch",
                    logging_dir=f"{output_dir}/logs",
                    save_total_limit=1
                )
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    compute_metrics=lambda p: compute_metrics(p)
                )

            with st.spinner("Training... This may take a while"):
                trainer.train()
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)

            st.success("Training complete and model saved.")
            st.header("3. Evaluation Metrics")
            metrics = trainer.evaluate()
            # extract confusion matrix and pop it from metrics dict
            cm = metrics.pop('eval_confusion_matrix', None) or trainer.compute_metrics(
                (np.array([0]), np.array([0]))
            )['confusion_matrix']
            st.write({k: v for k, v in metrics.items() if k.startswith('eval_') or k in ['accuracy','precision','recall','f1']})
            plot_confusion_matrix(cm)

elif mode == "Predict Only":
    st.header("Load Trained Model for Prediction")
    model_dir = st.text_input("Model Directory", value="./results")
    if st.button("Load Model & Tokenizer"):
        with st.spinner("Loading..."):
            tokenizer, model = load_trained(model_dir)
        st.success("Model loaded.")

        text = st.text_area("Enter text to classify:", height=150)
        if st.button("Predict"):
            if not text.strip():
                st.warning("Please enter text.")
            else:
                with st.spinner("Predicting..."):
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, padding="max_length", max_length=512)
                    logits = model(**inputs).logits
                    pred = torch.argmax(logits, dim=-1).item()
                    label_map = {0: "Left", 1: "Center", 2: "Right"}
                    st.success(f"Predicted Leaning: **{label_map[pred]}**")
