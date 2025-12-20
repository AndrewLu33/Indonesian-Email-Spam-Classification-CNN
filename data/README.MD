# 📂 Dataset Description

This folder contains datasets used for the Indonesian Email Spam Classification project.

---

## 📁 Folder Structure
```
data/
├── processed/
│   └── email_spam_clean.csv
├── raw/
│   └── email_spam_indo.csv
└── README.md
```

---

## 📄 Raw Dataset (`data/raw/`)

The raw dataset contains original email messages without any preprocessing.

**Columns:**
- `Pesan`    : Original email text
- `Kategori` : Label (`spam` or `ham`)

The raw dataset is preserved to ensure reproducibility and transparency.

---

## 🧹 Processed Dataset (`data/processed/`)

The processed dataset is generated after applying text preprocessing steps.

**Additional Column:**
- `clean_text` : Cleaned version of email text

**Preprocessing steps include:**
- Lowercasing
- URL and email removal
- Emoji removal
- Non-alphabetic character removal
- Stopword removal (Sastrawi + custom email stopwords)

This dataset is used as input for:
- TF-IDF + CNN model
- IndoBERT + CNN model (optional)

---

## ⚠️ Notes

- Raw data should not be modified.
- Processed data is regenerated using preprocessing scripts.
- The `processed` dataset should be used for model training.
