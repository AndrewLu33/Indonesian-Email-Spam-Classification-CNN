import re
import emoji
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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
    EMOJI_RE = emoji.get_emoji_regexp()
    strip_emoji = lambda t: EMOJI_RE.sub(" ", t)
else:
    strip_emoji = lambda t: emoji.replace_emoji(t, replace=" ")

def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = strip_emoji(text)
    text = NON_ALPHA.sub(" ", text)

    tokens = [
        t for t in text.split()
        if t not in STOPWORDS and len(t) > 2 and t.isalpha()
    ]
    return " ".join(tokens)

def save_cleaned_dataset(df, text_col, output_path):
    df["clean_text"] = df[text_col].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Cleaned dataset saved → {output_path}")
