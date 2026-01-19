from processor import loaded_df
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

def bar_plot():
    try:
        sentiment_counts = loaded_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']

        fig = px.bar(sentiment_counts, x='sentiment', y='count',
                    title='Distribution of Sentiment Categories',
                    labels={'sentiment': 'Sentiment', 'count': 'Number of Tweets'},
                    color='sentiment',
                    template='plotly_white')
        fig.update_layout(xaxis_title="Sentiment Category", yaxis_title="Number of Tweets")
        fig.show()
        print("Sentiment distribution plot generated.")

    except Exception as e:
        print(f"Error generating sentiment distribution plot: {e}")


# **Explanation of Sentiment Distribution Plot**:
# The bar chart above visually represents the number of tweets falling into each sentiment category. We can immediately observe the distribution of sentiments. For instance, if one sentiment (e.g., 'neutral' or 'worry') significantly outweighs others, it indicates a class imbalance. This information is vital as imbalanced datasets can lead to models that perform well on the majority class but poorly on minority classes. Our current dataset shows a potentially high number of 'neutral', 'sadness', and 'worry' tweets, while 'love' and 'happiness' are less frequent. The model will need to account for this imbalance to perform robustly across all sentiments.

# ### Tweet Length Distribution (Cleaned Content)
# This histogram shows the distribution of the length of the cleaned tweet content. This helps us understand typical tweet lengths and identify any unusual patterns.

def histogram_plot():

    try:
        fig = px.histogram(loaded_df, x='cleaned_content_length',
                        title='Distribution of Cleaned Tweet Lengths',
                        labels={'cleaned_content_length': 'Cleaned Content Length (characters)'},
                        nbins=50,
                        template='plotly_white')
        fig.update_layout(xaxis_title="Cleaned Content Length", yaxis_title="Number of Tweets")
        fig.show()
        print("Cleaned tweet length distribution plot generated.")

    except Exception as e:
        print(f"Error generating cleaned tweet length distribution plot: {e}")


# **Explanation of Tweet Length Distribution Plot**:
# The histogram above displays the frequency of different lengths of cleaned tweet content. We can observe that most tweets fall within a certain character range, typically short. This distribution helps us understand the typical verbosity of tweets after preprocessing. For instance, a long tail towards higher lengths might indicate some exceptionally detailed tweets, while a peak at lower lengths confirms the concise nature of tweets. This visualization helps confirm that our text cleaning process has produced reasonable text lengths and that there are no extreme outliers in length that could skew our model training (though such extreme outliers, if present, would have been handled during the 'outlier handling' phase of preprocessing).

# ### Top N-grams Word Cloud
# A word cloud visually represents the most frequent words in the entire corpus. This gives a quick overview of the most common themes or topics discussed in the tweets.

def word_cloud():


    try:
        all_words = ' '.join([text for text in loaded_df['cleaned_content']])
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(all_words)

        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.title('Word Cloud of Cleaned Tweet Content')
        plt.show()
        print("Word cloud generated.")

    except Exception as e:
        print(f"Error generating word cloud: {e}")


# **Explanation of Word Cloud**:
# The word cloud visually highlights the most frequently occurring words in our cleaned tweet dataset. Larger words in the cloud indicate higher frequency. This plot provides immediate insights into the predominant themes and vocabulary used across all tweets. For instance, words like "day," "good," "get," "go," "time," etc., are often prominent. This helps confirm the effectiveness of stop word removal (common words like "the", "is", "a" should be absent) and gives us an intuitive understanding of the overall content, which is useful for context but doesn't directly inform model architecture.

bar_plot()
histogram_plot()
word_cloud()