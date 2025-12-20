import streamlit as st
from model_loader import (
    load_tfidf_model, predict_tfidf,
    load_bert_model, predict_bert    
)

st.set_page_config(page_title="Email Spam Detection", layout="centered")

st.title("Email Spam Detection")
st.write("Aplikasi untuk mendeteksi apakah suatu email termasuk *SPAM* atau *NON-SPAM*.")

# =============================
# PILIHAN MODEL
# (IndoBERT masih ada, tapi nanti tidak dipakai)
# =============================

model_choice = st.selectbox(
    "Pilih Model Klasifikasi:",
    [
        "TF-IDF + CNN",
        "IndoBERT + CNN"   
    ]
)

# =============================
# LOAD MODEL
# =============================

try:
    if model_choice == "TF-IDF + CNN":
        vectorizer, tfidf_model, label_encoder = load_tfidf_model()

    elif model_choice == "IndoBERT + CNN":
        tokenizer, bert_model, label_encoder = load_bert_model()

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =============================
# INPUT EMAIL
# =============================

email_text = st.text_area("Masukkan teks email:")

if st.button("Prediksi"):
    if not email_text.strip():
        st.warning("⚠ Harap masukkan teks email terlebih dahulu.")
    else:
        try:
            # =========================================
            # TF-IDF
            # =========================================
            if model_choice == "TF-IDF + CNN":
                label = predict_tfidf(email_text, vectorizer, tfidf_model, label_encoder)

            # =========================================
            # IndoBERT 
            # =========================================
            elif model_choice == "IndoBERT + CNN":
                label = predict_bert(email_text, tokenizer, bert_model, label_encoder)

            # =========================================
            # HASIL OUTPUT
            # =========================================
            if label.lower() == "spam":
                st.error("🚨 Email ini terdeteksi sebagai **SPAM**!")
            else:
                st.success("✔ Email ini **AMAN** (Non-Spam).")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
