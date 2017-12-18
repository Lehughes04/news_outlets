# Dependencies
import tweepy
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Twitter API Keys
consumer_key = "##################################"
consumer_secret = "#######################################"
access_token = "##########################################"
access_token_secret = "##############################################"

# Twitter Credentials
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


#List to gather data
news_outlet_list = ['@BBCNews','@CBSNews','@CNN','@FoxNews','@nytimes']
news_outlet = []
tweet_user = []
tweet_text = []
tweet_date = []

#get tweet data
for target_term in news_outlet_list:
	public_tweets = api.search(target_term, count=5, result_type="recent")
	for tweet in public_tweets['statuses']:
		news_outlet = [target_term]
		tweet_user = tweet['user']['name']
		tweet_text = tweet['text']
		tweet_date = tweet['user']['created_at']

		
#create Dict of tweet data
tweets_dict = {'news_outlet':news_outlet,
			'username': tweet_user,
			'tweet_text': tweet_text,
			'tweet_date':tweet_date,
}

df_tweets = pd.DataFrame.from_dict(tweets_dict)

df_tweets['compound']=''
df_tweets['positive']=''
df_tweets['negative']=''
df_tweets['neutral']=''

df_tweets.head()


# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

#add sentiment scores to dataframe
for index,row in df_tweets.iterrows():

    # Run Vader Analysis on each tweet
    compound = analyzer.polarity_scores(row["text"])["compound"]
    pos = analyzer.polarity_scores(row["text"])["pos"]
    neu = analyzer.polarity_scores(row["text"])["neu"]
    neg = analyzer.polarity_scores(row["text"])["neg"]

    # Add each value to the appropriate array
    row['compound'].append(compound)
    row['positive'].append(pos)
    row['negative'].append(neg)
    row['neutral'].append(neu)
	
df_tweets.head()

#Export dataframe to csv
df_tweets.to_csv('News_Tweets_Data.csv',index=False)
