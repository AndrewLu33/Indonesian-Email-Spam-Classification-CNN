import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense

# ============================================================
# TF-IDF Vectorizer
# ============================================================

def build_tfidf_vectorizer(max_features=5000):
    return TfidfVectorizer(max_features=max_features)

def transform_tfidf(vectorizer, texts):
    tfidf_matrix = vectorizer.transform(texts).toarray()
    return np.expand_dims(tfidf_matrix, axis=2)   # CNN input: (None, 5000, 1)

# ============================================================
# Build CNN Model
# ============================================================

def build_tfidf_cnn(num_classes):
    model = Sequential([
        Conv1D(128, 5, activation='relu', input_shape=(5000, 1)),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
