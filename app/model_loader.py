import os
import re
import emoji
import joblib
import numpy as np
import tensorflow as tf

from transformers import BertTokenizer, TFBertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ======================================================================
# BASE DIRECTORY — gunakan output folder di luar src/
# ======================================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # /src
BASE_DIR = os.path.dirname(SRC_DIR)                           # project root
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")       # /outputs/models


# ======================================================================
# PATHS
# ======================================================================
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, "model_tfidf_cnn.h5")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

BERT_TOKENIZER_DIR = os.path.join(MODEL_DIR, "bert_tokenizer")
BERT_MODEL_DIR = os.path.join(MODEL_DIR, "model_bert_cnn")


# ======================================================================
# LOAD TF-IDF (TOKENIZER + CNN) MODEL
# ======================================================================
def load_tfidf_model():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = tf.keras.models.load_model(TFIDF_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return vectorizer, model, label_encoder

# ======================================================================
# LOAD BERT MODEL
# ======================================================================
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_DIR)

    model = tf.keras.models.load_model(
        BERT_MODEL_DIR,
        compile=False,
        custom_objects={"TFBertModel": TFBertModel}
    )

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return tokenizer, model, label_encoder


# ======================================================================
# CLEANING FUNCTION
# ======================================================================
STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
STOPWORDS.update({
    "hou","kaminski","vince","enron","corp","edu","cc","re","fw",
    "subject","email","houston","pm","am","com","net","org",
    "ltd","co","inc","ect"
})

URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
NON_ALPHA = re.compile(r'[^a-zA-Z\s]')

if hasattr(emoji, "get_emoji_regexp"):
    strip_emoji = lambda txt: emoji.get_emoji_regexp().sub(" ", txt)
else:
    strip_emoji = lambda txt: emoji.replace_emoji(txt, replace=" ")


def clean_text(text):
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = strip_emoji(text)
    text = NON_ALPHA.sub(" ", text)

    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2 and t.isalpha()]
    return " ".join(tokens)



# ======================================================================
# TF-IDF (TOKENIZER SEQUENCE CNN) PREDICT
# ======================================================================
def predict_tfidf(text, vectorizer, model, label_encoder):
    clean = clean_text(text)
    tfidf = vectorizer.transform([clean]).toarray()
    tfidf = np.expand_dims(tfidf, axis=2)
    pred = model.predict(tfidf).argmax(axis=1)[0]
    return label_encoder.inverse_transform([pred])[0]




# ======================================================================
# BERT PREDICT
# ======================================================================
def predict_bert(text, tokenizer, model, label_encoder):
    print("DEBUG predict_bert input:", type(text), text)

    clean = clean_text(text)

    enc = tokenizer(
        [clean],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="tf"
    )


    pred = model.predict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"]
    })

    pred = pred.argmax(axis=1)[0]
    return label_encoder.inverse_transform([pred])[0]

