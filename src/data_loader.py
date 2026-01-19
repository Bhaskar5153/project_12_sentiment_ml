import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    return df


loaded_df = load_data(file_path=r"data\tweet_emotions.csv")
# print(loaded_df)