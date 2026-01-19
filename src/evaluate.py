from processor import loaded_df
from models import model_results
from models import X_test, X_train, y_test, y_train, X, y
from sklearn.metrics import confusion_matrix
import plotly.express as px
import pandas as pd
from processor import clean_text


print("Displaying detailed evaluation metrics for each model.")

for name, results in model_results.items():
    print(f"\n--- Detailed Evaluation for {name} ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (weighted): {results['f1_score']:.4f}")
    
    # Plot Confusion Matrix
    y_pred = results['model'].predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=y.unique())
    
    fig_cm = px.imshow(cm,
                       labels=dict(x="Predicted Sentiment", y="Actual Sentiment", color="Count"),
                       x=y.unique(),
                       y=y.unique(),
                       color_continuous_scale='Viridis',
                       title=f'Confusion Matrix for {name}')
    fig_cm.update_xaxes(side="bottom")
    fig_cm.show()
    print(f"Confusion Matrix for {name} displayed.")


# **Explanation of Confusion Matrices**:
# The confusion matrices above provide a detailed breakdown of each model's performance. Each matrix is a square table where:
# *   **Rows** represent the actual sentiment categories.
# *   **Columns** represent the sentiment categories predicted by the model.
# *   **Diagonal values** (from top-left to bottom-right) indicate the number of correct predictions for each class (True Positives).
# *   **Off-diagonal values** indicate misclassifications. For example, a value in row 'sadness' and column 'neutral' means the model incorrectly predicted a 'sadness' tweet as 'neutral'.

# **Interpretation**:
# By examining the confusion matrix for each model, we can:
# *   **Identify well-predicted classes**: Classes with high values along the diagonal.
# *   **Identify problematic classes**: Classes with low diagonal values and high off-diagonal values, indicating frequent misclassification.
# *   **Understand common confusions**: See which classes are often mistaken for others (e.g., 'worry' often confused with 'sadness' or 'neutral'). This gives insights into the ambiguity between certain sentiments.

# For instance, we might observe that `neutral` and `worry` are often confused by the models, which might be due to their textual content being similar or ambiguous. `love` or `hate` might be easier to distinguish due to more distinct keywords. This detailed view helps in fine-tuning model strategies or improving feature engineering for specific sentiment pairs.



print("Creating example dataset and making predictions.")

# Select the best model (e.g., based on F1-score)
best_model_name = max(model_results, key=lambda name: model_results[name]['f1_score'])
best_pipeline = model_results[best_model_name]['model']
print(f"Using the best model for example predictions: {best_model_name}")

# Create a sample dataset of new tweets
sample_tweets = pd.DataFrame({
    'tweet_id': [900000001, 900000002, 900000003, 900000004, 900000005],
    'content': [
        "I'm so happy today! What a wonderful day!",
        "This is really making me angry. I hate it.",
        "Just woke up, feeling neutral about everything.",
        "I'm worried about the upcoming exam. Fingers crossed.",
        "@user Check out this cool new product! #innovation"
    ]
})

print("Sample tweets created:")
print(sample_tweets.to_string())

try:
    # Preprocess the sample tweets using the same cleaning function
    sample_tweets['cleaned_content'] = sample_tweets['content'].apply(clean_text)
    print("Sample tweets preprocessed.")

    # Make predictions using the best model
    predictions = best_pipeline.predict(sample_tweets['cleaned_content'])
    sample_tweets['predicted_sentiment'] = predictions

    print("Predictions made on sample tweets:")
    print(sample_tweets[['content', 'predicted_sentiment']].to_string())

except Exception as e:
    print(f"An error occurred while making predictions on sample data: {e}")
