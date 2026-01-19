from processor import loaded_df
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score


try:
    X = loaded_df['cleaned_content'] # Features are the cleaned tweet content
    y = loaded_df['sentiment']       # Target is the sentiment label

    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    print(f"Sample X (first 5):\n{X.head().to_string()}")
    print(f"Sample y (first 5):\n{y.head().to_string()}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into training and testing sets.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Check the distribution of sentiments in train and test sets (due to stratify=y)
    print(f"Sentiment distribution in y_train:\n{y_train.value_counts(normalize=True).to_string()}")
    print(f"Sentiment distribution in y_test:\n{y_test.value_counts(normalize=True).to_string()}")

except Exception as e:
    print(f"An error occurred during feature separation and data splitting: {e}")


print("Starting model training with different classifiers.")

# Define the models to be tested
models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='saga'),
    'Support Vector Machine': LinearSVC(class_weight="balanced") # probability=True for consistent evaluation
}

# Store results for comparison
model_results = {}

# Iterate over each model
for name, model in models.items():
    print(f"Training {name} model...")
    try:
        # Create a pipeline with TfidfVectorizer and the classifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7)), # max_features, min_df, max_df for feature selection (handling rare/common words)
            ('classifier', model)
        ])

        # Train the model
        pipeline.fit(X_train, y_train)
        print(f"{name} model trained successfully.")

        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)

        model_results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }

        print(f"--- {name} Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    except Exception as e:
        print(f"Error training or evaluating {name} model: {e}")

print("All models trained and evaluated.")

