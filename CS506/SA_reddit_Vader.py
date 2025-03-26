import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load Reddit data
df = pd.read_csv("final_output.csv", header=None)
df.columns = ['score', 'date', 'user', 'url', 'comment']

# Initialize VADER
vader = SentimentIntensityAnalyzer()

# Analyze sentiment using VADER
def get_vader_sentiment(text):
    try:
        scores = vader.polarity_scores(str(text))
        compound = scores['compound']
        if compound >= 0.05:
            label = 'POSITIVE'
        elif compound <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        return pd.Series([compound, label])
    except:
        return pd.Series([None, None])

# Apply to comments
df[['vader_compound', 'vader_label']] = df['comment'].apply(get_vader_sentiment)

# Save the results
df.to_csv("reddit_comments_with_vader.csv", index=False)
print("File saved as 'reddit_comments_with_vader.csv'")

