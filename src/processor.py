import nltk
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
import re
import string
from data_loader import load_data

loaded_df = load_data(r"data\tweet_emotions.csv")



# Initialize NLTK tools
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a text cleaning function
def clean_text(text):
    if not isinstance(text, str): # Handle potential non-string types
        return ""
    text = text.lower() # Lowercase the text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words) # Remove stopwords
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split()) # Lemmatize words
    return text


loaded_df['cleaned_content'] = loaded_df['content'].apply(clean_text)
# print(loaded_df)
print(loaded_df.shape[0])

loaded_df = loaded_df[loaded_df['cleaned_content'].str.strip() != '']
# print(loaded_df)
print(loaded_df.shape[0])

loaded_df.drop('content', axis=1, inplace=True)

# Outlier handling for tweet length (conceptual for text data)
# We can define "outliers" as extremely short or long tweets.
# The `content_length` was based on original content. Let's calculate for cleaned content.
loaded_df['cleaned_content_length'] = loaded_df['cleaned_content'].apply(len)


# A simple approach to 'outliers' in length: remove tweets with extremely short cleaned content,
#     # as they might not carry enough information for sentiment classification.
#     # Threshold can be empirically chosen or based on distribution.
min_length_threshold = 5 # Example: remove tweets with less than 5 characters after cleaning
original_shape_post_cleaning = loaded_df.shape[0]
loaded_df = loaded_df[loaded_df['cleaned_content_length'] >= min_length_threshold]
removed_short_tweets = original_shape_post_cleaning - loaded_df.shape[0]
if removed_short_tweets > 0:
    print(f"Removed {removed_short_tweets} tweets with cleaned content length less than {min_length_threshold}. New shape: {loaded_df.shape}")
else:
    print("No tweets removed based on minimum cleaned content length.")

