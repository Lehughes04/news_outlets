
### Observations

1.  The BBC News was the only outlet to have a positive average compound sentiment analysis.
2.  CNN had the most negative average compound sentiment analysis of all the news outlets.
3.  The New York times had, on average, the most neutral sentiment over the last 100 tweets.


```python
# Dependencies
import tweepy
import time
import datetime
import json
import pandas as pd
import numpy as np
```


```python
#Twitter API Keys
consumer_key = "#########################"
consumer_secret = "####################################"
access_token = "###########################"
access_token_secret = "###################################"

# Twitter Credentials
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
#List to gather data
news_outlet_list = ['@BBCNews','@CBSNews','@CNN','@FoxNews','@nytimes']
news_outlet = []
tweet_user = []
tweet_text = []
tweet_date = []
```


```python
#get tweet data
for news in news_outlet_list:
	for x in range(5):
		public_tweets = api.user_timeline(news, count=20, page=x)
		for tweet in public_tweets:
			news_outlet.append(news)
			tweet_user.append(tweet['user']['screen_name'])
			tweet_text.append(tweet['text'])
			tweet_date.append(tweet['created_at'])
			#time.sleep(5)

len(tweet_text)
```




    500




```python
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

df_tweets.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>news_outlet</th>
      <th>tweet_date</th>
      <th>tweet_text</th>
      <th>username</th>
      <th>compound</th>
      <th>positive</th>
      <th>negative</th>
      <th>neutral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCNews</td>
      <td>Mon Dec 18 01:49:35 +0000 2017</td>
      <td>Depression: 'I kept my head down to survive th...</td>
      <td>BBCNews</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCNews</td>
      <td>Mon Dec 18 01:38:59 +0000 2017</td>
      <td>Steroid abuse 'raising health risk for thousan...</td>
      <td>BBCNews</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCNews</td>
      <td>Sun Dec 17 22:18:02 +0000 2017</td>
      <td>The Apprentice: Lord Sugar surprises viewers w...</td>
      <td>BBCNews</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCNews</td>
      <td>Sun Dec 17 21:32:20 +0000 2017</td>
      <td>BBC Sports Personality of the Year 2017: World...</td>
      <td>BBCNews</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCNews</td>
      <td>Sun Dec 17 21:06:29 +0000 2017</td>
      <td>RT @BBCSport: He won his third consecutive 10,...</td>
      <td>BBCNews</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

#add sentiment scores to dataframe
for index,row in df_tweets.iterrows():

    # Run Vader Analysis on each tweet
    compound = analyzer.polarity_scores(row["tweet_text"])["compound"]
    pos = analyzer.polarity_scores(row["tweet_text"])["pos"]
    neu = analyzer.polarity_scores(row["tweet_text"])["neu"]
    neg = analyzer.polarity_scores(row["tweet_text"])["neg"]

    # Add each value to the appropriate array
    row['compound'] = compound
    row['positive'] = pos
    row['negative'] = neg
    row['neutral'] = neu

#Add date field and sort descending
df_tweets['date'] = pd.to_datetime(df_tweets['tweet_date'])
df_tweets_sort = df_tweets.sort_values(by='date',ascending=False).reset_index(drop=True)
df_tweets_sort.head()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>news_outlet</th>
      <th>tweet_date</th>
      <th>tweet_text</th>
      <th>username</th>
      <th>compound</th>
      <th>positive</th>
      <th>negative</th>
      <th>neutral</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@nytimes</td>
      <td>Mon Dec 18 02:32:12 +0000 2017</td>
      <td>Toronto Buzzes With Talk of Billionaire Couple...</td>
      <td>nytimes</td>
      <td>-0.7506</td>
      <td>0</td>
      <td>0.444</td>
      <td>0.556</td>
      <td>2017-12-18 02:32:12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@nytimes</td>
      <td>Mon Dec 18 02:32:12 +0000 2017</td>
      <td>Toronto Buzzes With Talk of Billionaire Couple...</td>
      <td>nytimes</td>
      <td>-0.7506</td>
      <td>0</td>
      <td>0.444</td>
      <td>0.556</td>
      <td>2017-12-18 02:32:12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@FoxNews</td>
      <td>Mon Dec 18 02:23:34 +0000 2017</td>
      <td>#ALSen.-Elect @GDouglasJones Calls for 'Common...</td>
      <td>FoxNews</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2017-12-18 02:23:34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@FoxNews</td>
      <td>Mon Dec 18 02:22:14 +0000 2017</td>
      <td>.@KevinJacksonTBS: "They say nothing is foolpr...</td>
      <td>FoxNews</td>
      <td>-0.1984</td>
      <td>0.154</td>
      <td>0.238</td>
      <td>0.608</td>
      <td>2017-12-18 02:22:14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@FoxNews</td>
      <td>Mon Dec 18 02:22:14 +0000 2017</td>
      <td>.@KevinJacksonTBS: "They say nothing is foolpr...</td>
      <td>FoxNews</td>
      <td>-0.1984</td>
      <td>0.154</td>
      <td>0.238</td>
      <td>0.608</td>
      <td>2017-12-18 02:22:14</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Export dataframe to csv
df_tweets_sort.to_csv('News_Tweets_Data.csv',index=False)
```


```python
#Create Scatter plot of each tweet's compound sentiment. 
#['@BBCNews','@CBSNews','@CNN','@FoxNews','@nytimes']
import matplotlib.pyplot as plt
import seaborn as sns

df_bbc = df_tweets_sort.loc[df_tweets['news_outlet']=='@BBCNews']
df_cbs = df_tweets_sort.loc[df_tweets['news_outlet']=='@CBSNews']
df_cnn = df_tweets_sort.loc[df_tweets['news_outlet']=='@CNN']
df_fox = df_tweets_sort.loc[df_tweets['news_outlet']=='@FoxNews']
df_nyt = df_tweets_sort.loc[df_tweets['news_outlet']=='@nytimes']

plt.scatter(range(len(df_bbc)),df_bbc['compound'],color = 'blue',label='BBC')
plt.scatter(range(len(df_cbs)),df_cbs['compound'],color = 'red',label='CBS')
plt.scatter(range(len(df_cnn)),df_cnn['compound'],color = 'green',label='CNN')
plt.scatter(range(len(df_fox)),df_fox['compound'],color = 'yellow',label='FOXNews')
plt.scatter(range(len(df_nyt)),df_nyt['compound'],color = 'black',label='NYTimes')

plt.xlabel('Number of Tweets Ago')
plt.ylabel('Tweet Sentiment Magnitude')
plt.ylim(-1.1,1.1)
plt.xlim(0,130)
plt.title('Sentiment Analysis of News Media Tweets (2017-12-17)')
plt.legend(loc='best',fancybox=True)

plt.savefig('News-Outlet-Sentiment-Scatter')
plt.show()

```


![png](output_8_0.png)



```python
#Bar chart showing average compound sentiment for each news org

x = np.arange(5)

df_grp = df_tweets_sort
df_grp['compound'] = df_grp['compound'].apply(pd.to_numeric)

df_grp_mean = df_grp.groupby('news_outlet')
df_grp_mean_final = pd.DataFrame(df_grp_mean['compound'].mean())
#df_grp_mean_final = df_grp_mean_final.reset_index()
df_grp_mean_final.head()

barlist = plt.bar(x,df_grp_mean_final['compound'])
plt.xticks(x,df_grp_mean_final.index)
barlist[0].set_color('blue')
barlist[1].set_color('red')
barlist[2].set_color('green')
barlist[3].set_color('yellow')
barlist[4].set_color('black')
plt.ylim(-.2,.1)
plt.ylabel('Sentiment Score')
plt.xlabel('News Outlets')
plt.title("Average Sentiment for News Outlet's Last 100 Tweets on 2017-12-17")
plt.savefig('News-Outlet-Mean-Sentiment-Bar')
plt.show()
```


![png](output_9_0.png)



```python

```
