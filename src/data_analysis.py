from data_loader import load_data
import pandas as pd
import numpy as np
import logging


loaded_df = load_data(file_path=r"data\tweet_emotions.csv")
# print(loaded_df)
# check for null values


# check for unique tweet ids
unique_tweet_id = loaded_df['tweet_id'].nunique()
# print(unique_tweet_id)
# print(loaded_df.info())
total_tweets = len(loaded_df)

# print(unique_tweet_id, total_tweets)

sentiment_count = loaded_df['sentiment'].value_counts()
# print(sentiment_count)
content_length = loaded_df['content'].apply(len)
print(content_length)
print(loaded_df.info())

