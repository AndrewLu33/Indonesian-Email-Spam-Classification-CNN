import streamlit as st
from model_loader import (
    load_tfidf_model, predict_tfidf,
    load_bert_model, predict_bert
)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Email Spam Detection",
    layout="centered"
)

st.title("📧 Email Spam Detection")
st.write(
    "Aplikasi untuk mendeteksi apakah suatu email termasuk **SPAM** "
    "atau **NON-SPAM** menggunakan model Machine Learning."
)

# ======================================================
# LAZY + CACHED MODEL LOADERS
# ======================================================
@st.cache_resource(show_spinner="🔄 Memuat model TF-IDF + CNN...")
def get_tfidf_model():
    return load_tfidf_model()


@st.cache_resource(show_spinner="🔄 Memuat model IndoBERT + CNN (pertama kali agak lama)...")
def get_bert_model():
    return load_bert_model()


# ======================================================
# MODEL SELECTION
# ======================================================
model_choice = st.selectbox(
    "Pilih Model Klasifikasi:",
    [
        "TF-IDF + CNN",
        "IndoBERT + CNN"
    ]
)

# ======================================================
# INPUT EMAIL
# ======================================================
email_text = st.text_area(
    "Masukkan teks email:",
    height=200,
)

# ======================================================
# PREDICTION
# ======================================================
if st.button("🔍 Prediksi"):
    if not email_text.strip():
        st.warning("⚠ Harap masukkan teks email terlebih dahulu.")
        st.stop()

    try:
        # ===============================
        # TF-IDF MODEL
        # ===============================
        if model_choice == "TF-IDF + CNN":
            vectorizer, tfidf_model, label_encoder = get_tfidf_model()
            label = predict_tfidf(
                email_text,
                vectorizer,
                tfidf_model,
                label_encoder
            )

        # ===============================
        # BERT MODEL
        # ===============================
        elif model_choice == "IndoBERT + CNN":
            tokenizer, bert_model, label_encoder = get_bert_model()
            label = predict_bert(
                email_text,
                tokenizer,
                bert_model,
                label_encoder
            )

        # ===============================
        # OUTPUT
        # ===============================
        if label.lower() == "spam":
            st.error("🚨 Email ini terdeteksi sebagai **SPAM**!")
        else:
            st.success("✅ Email ini **AMAN** (Non-Spam).")

    except Exception as e:
        st.error("❌ Terjadi error saat prediksi.")
        st.exception(e)
