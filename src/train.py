from sklearn.model_selection import train_test_split
from joblib import dump
from src.data_loader import load_data
from src.processor import preprocess_texts
from src.models import build_model
from src.evaluate import evaluate_model

def train_pipeline(data_path="data/tweet_emotions.csv", model_type="svm"):
    X, y = load_data(data_path)
    X_clean = preprocess_texts(X.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(model_type)
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    dump(model, f"artifacts/{model_type}_model.joblib")
    print(f"Model saved to artifacts/{model_type}_model.joblib")

if __name__ == "__main__":
    train_pipeline()