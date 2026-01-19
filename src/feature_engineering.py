import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def text_length(texts):
    return pd.DataFrame([len(t.split()) for t in texts])

def build_features(train_texts, test_texts, max_features=8000):
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=max_features,
        ngram_range=(1,2)
    )
    length_feat = FunctionTransformer(lambda x: text_length(x), validate=False)

    combined = FeatureUnion([
        ("tfidf", tfidf),
        ("length", length_feat)
    ])

    X_train = combined.fit_transform(train_texts)
    X_test = combined.transform(test_texts)
    return X_train, X_test, combined