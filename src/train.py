import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import save_cleaned_dataset
from data_loader import load_data, split_data
from model_tfidf import build_tfidf_vectorizer, transform_tfidf, build_tfidf_cnn
from model_bert import encode_bert, build_bert_cnn
from evaluate import evaluate_model, plot_side_by_side

import numpy as np
import os
from transformers import BertTokenizer



# ============================================================
# BASE PATHS (BENAR – POINTING TO PROJECT ROOT)
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))              # src/
BASE_DIR = os.path.dirname(SRC_DIR)                               # Final Project Deep Learning/

DATA_DIR = os.path.join(BASE_DIR, "data")                         # Final Project/data
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")                    # Final Project/outputs
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")                    # Final Project/outputs/models
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")                        # Final Project/outputs/logs



# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)



# ============================================================
# LOAD & CLEAN DATA
# ============================================================
df = load_data(os.path.join(DATA_DIR, "raw", "email_spam_indo.csv"))

save_cleaned_dataset(
    df,
    "Pesan",
    os.path.join(DATA_DIR, "processed", "email_spam_clean.csv")
)

df = pd.read_csv(os.path.join(DATA_DIR, "processed", "email_spam_clean.csv"))



# ============================================================
# LABEL ENCODING (SAVE)
# ============================================================
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["Kategori"])
labels = le.classes_

joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))



# ============================================================
# SPLIT DATA
# ============================================================
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)



# ============================================================
# TF-IDF TOKEN-BASED CNN
# ============================================================


vectorizer = build_tfidf_vectorizer()
vectorizer.fit(X_train)

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# Transform TF-IDF → CNN Input
X_train_cnn = transform_tfidf(vectorizer, X_train)
X_val_cnn   = transform_tfidf(vectorizer, X_val)
X_test_cnn  = transform_tfidf(vectorizer, X_test)

model_tfidf = build_tfidf_cnn(num_classes=len(labels))

history_tfidf = model_tfidf.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# SAVE MODEL
model_tfidf.save(os.path.join(MODEL_DIR, "model_tfidf_cnn.h5"))

# EVALUATE
pred_tfidf = model_tfidf.predict(X_test_cnn).argmax(axis=1)

evaluate_model("TF-IDF + CNN", y_test, pred_tfidf, labels)

# ============================================================
# BERT CNN MODEL
# ============================================================
bert_tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
bert_tokenizer.save_pretrained(os.path.join(MODEL_DIR, "bert_tokenizer"))

train_enc = encode_bert(X_train)
val_enc = encode_bert(X_val)
test_enc = encode_bert(X_test)

model_bert = build_bert_cnn(num_classes=len(labels))

history_bert = model_bert.fit(
    [train_enc["input_ids"], train_enc["attention_mask"]],
    y_train,
    validation_data=(
        [val_enc["input_ids"], val_enc["attention_mask"]],
        y_val
    ),
    epochs=10,
    batch_size=16
)

model_bert.save(os.path.join(MODEL_DIR, "model_bert_cnn"))

pred_bert = model_bert.predict([
    test_enc["input_ids"], test_enc["attention_mask"]
]).argmax(axis=1)

evaluate_model("IndoBERT + CNN", y_test, pred_bert, labels)

plot_side_by_side(history_tfidf, history_bert)
