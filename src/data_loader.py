import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, text_col="Pesan", label_col="Kategori"):
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    return df

def split_data(df, clean_col="clean_text", label_col="label_enc"):
    X = df[clean_col]
    y = df[label_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
