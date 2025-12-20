## 📌 Project Summary

This project investigates the effectiveness of **deep learning approaches for Indonesian email spam detection**, with a focus on how different **text representation strategies** influence classification performance. Email spam remains a persistent digital threat, often involving phishing, fraud, and misleading promotions that can lead to financial loss and security breaches. Due to the evolving and deceptive nature of spam content, traditional rule-based filtering systems have become increasingly ineffective.

To address this challenge, this study compares two deep learning pipelines:
- **TF-IDF + Convolutional Neural Network (CNN)**
- **IndoBERT + Convolutional Neural Network (CNN)**

The objective is to analyze the trade-offs between **computational efficiency, model complexity, and classification accuracy** in the context of Indonesian-language spam detection.

---

## 📂 Dataset

The experiments were conducted using the **Indonesian Email Spam** dataset obtained from Kaggle. The dataset consists of labeled email texts categorized into **spam** and **non-spam (ham)** classes. It is a translated and localized version of an English spam dataset, adapted to reflect Indonesian linguistic characteristics and common spam patterns.

Before modeling, the dataset underwent extensive preprocessing, including text normalization, removal of URLs, emails, emojis, punctuation, stopwords, and stemming. This ensured data consistency and improved model learning efficiency.

---

## 🧠 Methodology

The project follows a structured machine learning workflow consisting of data preprocessing, exploratory data analysis, feature extraction, model training, and evaluation.

- **TF-IDF + CNN**  
  TF-IDF transforms email text into sparse numerical vectors based on word importance across the corpus. These vectors are reshaped and passed into a 1D CNN to learn local n-gram patterns indicative of spam behavior. This approach serves as a lightweight and computationally efficient baseline.

- **IndoBERT + CNN**  
  IndoBERT, a transformer-based language model pretrained on large Indonesian corpora, generates contextual embeddings that capture semantic and syntactic relationships between words. These embeddings are further processed by a CNN to extract salient local features before classification. This hybrid architecture combines global contextual understanding with local pattern detection.

The dataset was split into training, validation, and testing sets using an **80:10:10** ratio with stratified sampling to maintain class balance.

---

## 📊 Evaluation & Results

Model performance was evaluated using **Accuracy, Precision, Recall, and F1-score** on an unseen test set.

| Model            | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| TF-IDF + CNN     | 0.5909   | 0.5971    | 0.5943 | 0.5890   |
| IndoBERT + CNN   | 0.9810   | 0.9809    | 0.9811 | 0.9810   |

The results clearly demonstrate that **IndoBERT + CNN significantly outperforms TF-IDF + CNN** across all evaluation metrics. While the TF-IDF-based model is able to capture keyword-driven patterns, it struggles with semantic ambiguity and contextual understanding. In contrast, IndoBERT’s contextual embeddings enable rapid convergence, superior generalization, and much higher classification accuracy.

---

## 🚀 Deployment

Both trained models were deployed using a **Streamlit-based web application**, allowing users to input raw email text and receive real-time spam classification results. The deployment pipeline preserves consistency with the training process by reusing the same preprocessing steps, vectorizers, and tokenizers.

Although currently deployed locally, the application can be easily extended to cloud platforms such as **Heroku, Render, or Vercel**, demonstrating its readiness for real-world usage.

---

## 🧩 Key Takeaways

- **TF-IDF + CNN** is suitable as a lightweight baseline with low computational cost.
- **IndoBERT + CNN** provides state-of-the-art performance for Indonesian spam detection due to its contextual understanding.
- Text representation plays a critical role in determining deep learning model performance.
- Transformer-based models are especially effective for languages with rich morphology and informal variations like Indonesian.

---

## 🎓 Authors

- **Andrew** (2702286715)  
- **Nicholas Wira Angkasa** (2702294521)  

Computer Science Department  
School of Computer Science  
Bina Nusantara University, Jakarta, Indonesia

---

## 🎥 Demo Video

A full demonstration of the application, including model prediction and workflow, can be viewed here:

🔗 **YouTube Demo**  
https://www.youtube.com/watch?v=uja_BXn9UUo

---

## 📊 Presentation Slides

The complete project presentation, including methodology, model architecture, results, and analysis, is available at:

🔗 **Canva Presentation**  
https://www.canva.com/design/DAG7TsLM2OA/3fmho2B5dqeHWfCU0rguyQ/edit?utm_content=DAG7TsLM2OA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

