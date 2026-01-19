import joblib
from src.data_loader import load_data
from src.feature_engineering import build_features
from src.train import train_models
from src.evaluate import evaluate_model
from src.visualize import plot_feature_importance

def main():
    X_train, X_test, y_train, y_test = load_data("data/tweet_emotions.csv")
    X_train_vec, X_test_vec, vectorizer = build_features(X_train, X_test)

    joblib.dump(vectorizer, "models/vectorizer.pkl")

    models = train_models(X_train_vec, y_train)

    for name, model in models.items():
        evaluate_model(model, X_test_vec, y_test, name)
        plot_feature_importance(vectorizer, model)

if __name__ == "__main__":
    main()