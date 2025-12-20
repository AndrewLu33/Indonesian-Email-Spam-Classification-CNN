import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# ============================================================
# BASE DIRECTORY (parent folder dari src/)
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ============================================================
# Pilih colormap berdasarkan model
# ============================================================
def get_heatmap_color(model_name):
    name = model_name.lower()
    if "tf-idf" in name or "tfidf" in name:
        return "Reds"
    return "Blues"


# ============================================================
# Confusion Matrix Heatmap
# ============================================================
def plot_confusion_heatmap(y_true, y_pred, labels, title, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cmap = get_heatmap_color(model_name)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


# ============================================================
# Training Curves → disimpan ke outputs/plots/
# ============================================================
def plot_training_curves(history, model_name):
    ensure_dir(LOG_DIR)

    # Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title(f"{model_name} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_accuracy.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{model_name} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_loss.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Evaluate model → logs (report + heatmap)
# ============================================================
def evaluate_model(model_name, y_true, y_pred, labels):
    ensure_dir(LOG_DIR)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {model_name} ACCURACY: {acc:.4f} ===\n")

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    report_path = os.path.join(LOG_DIR, f"{model_name}_classification_report.csv")
    df_report.to_csv(report_path)
    print(f"[SAVED] Classification Report → {report_path}")

    # Heatmap
    heatmap_path = os.path.join(LOG_DIR, f"{model_name}_heatmap.png")
    plot_confusion_heatmap(
        y_true, y_pred, labels,
        title=f"Confusion Matrix - {model_name}",
        model_name=model_name,
        save_path=heatmap_path
    )
    print(f"[SAVED] Heatmap → {heatmap_path}")

    return acc, df_report


# ============================================================
# Combined evaluation function
# ============================================================
def full_evaluation(model_name, history, y_true, y_pred, labels):
    ensure_dir(LOG_DIR)

    # Save training curves
    plot_training_curves(history, model_name)

    # Save logs (classification report + heatmap)
    acc, report = evaluate_model(model_name, y_true, y_pred, labels)

    print(f"\n==== Evaluation for {model_name} DONE ====\n")
    return acc, report


# ============================================================
# Side-by-Side Comparison (Loss & Accuracy)
# ============================================================
def plot_side_by_side(history_tfidf, history_bert):
    ensure_dir(LOG_DIR)

    # -------------------- LOSS COMPARISON --------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TF-IDF Loss
    axes[0].plot(history_tfidf.history["loss"], label="Train Loss", color="red")
    axes[0].plot(history_tfidf.history["val_loss"], label="Val Loss", color="darkred")
    axes[0].set_title("TF-IDF + CNN Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # IndoBERT Loss
    axes[1].plot(history_bert.history["loss"], label="Train Loss", color="blue")
    axes[1].plot(history_bert.history["val_loss"], label="Val Loss", color="navy")
    axes[1].set_title("IndoBERT + CNN Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "tfidf_indobert_loss.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print("[SAVED] → outputs/logs/tfidf_indobert_loss.png")

    # ------------------ ACCURACY COMPARISON ------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TF-IDF Accuracy
    axes[0].plot(history_tfidf.history["accuracy"], label="Train Accuracy", color="red")
    axes[0].plot(history_tfidf.history["val_accuracy"], label="Val Accuracy", color="darkred")
    axes[0].set_title("TF-IDF + CNN Accuracy Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # IndoBERT Accuracy
    axes[1].plot(history_bert.history["accuracy"], label="Train Accuracy", color="blue")
    axes[1].plot(history_bert.history["val_accuracy"], label="Val Accuracy", color="navy")
    axes[1].set_title("IndoBERT + CNN Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "tfidf_indobert_accuracy.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print("[SAVED] → outputs/logs/tfidf_indobert_accuracy.png")
