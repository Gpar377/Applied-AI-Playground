import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

# Reddit API credentials (replace with your own)
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
user_agent = 'reddit-sentiment-tracker'

reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

analyzer = SentimentIntensityAnalyzer()

# Choose subreddit and number of posts
subreddit_name = 'technology'
num_posts = 50

posts = reddit.subreddit(subreddit_name).hot(limit=num_posts)

data = []
for post in posts:
    title = post.title
    score = analyzer.polarity_scores(title)
    data.append({
        'title': title,
        'positive': score['pos'],
        'neutral': score['neu'],
        'negative': score['neg'],
        'compound': score['compound']
    })

df = pd.DataFrame(data)

# Plot compound sentiment scores
fig = px.histogram(df, x='compound', nbins=20, title=f'Sentiment Distribution in r/{subreddit_name}')
fig.show()
