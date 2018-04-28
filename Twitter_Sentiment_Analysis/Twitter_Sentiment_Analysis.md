

```python
#import dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

#import api keys
from config import (consumer_key,
                    consumer_secret,
                    access_token,
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target User Accounts
target_user = ("@bbc", "@cbs", "@cnn", "@foxnews", "@nytimes")

#list to store 100 tweets from each news outlet

news_source = []
compound = []
pos = []
neu = []
neg = []


tweets_ago = []
```


```python
for user in target_user:
    counter = 0
    for x in range(0, 5):
        # Get all tweets from home feed
            public_tweets = api.user_timeline(user, page=x)

            # Loop through all tweets
            for tweet in public_tweets:
                counter += 1
                # Run Vader Analysis on each tweet
                results = analyzer.polarity_scores(tweet["text"])
                news_source.append(user)
                compound.append(results["compound"])
                pos.append(results["pos"])
                neu.append(results["neu"])
                neg.append(results["neg"])
                tweets_ago.append(counter)
                
news_df = pd.DataFrame({"news_source" : news_source, "compound" : compound, "positive" : pos, "neutral": neu, "negative": neg, "tweets_ago": tweets_ago})
```


```python
news_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
      <th>negative</th>
      <th>neutral</th>
      <th>news_source</th>
      <th>positive</th>
      <th>tweets_ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6369</td>
      <td>0.25</td>
      <td>0.652</td>
      <td>@bbc</td>
      <td>0.098</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7650</td>
      <td>0.00</td>
      <td>0.680</td>
      <td>@bbc</td>
      <td>0.320</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>@bbc</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5994</td>
      <td>0.00</td>
      <td>0.786</td>
      <td>@bbc</td>
      <td>0.214</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>@bbc</td>
      <td>0.000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
mean_news_df = news_df.groupby("news_source").mean()["compound"]
```


```python
mean_news_df2 = pd.DataFrame(mean_news_df)
```


```python
mean_news_df2
bbc_avg = mean_news_df2[mean_news_df.index=="@bbc"].values[0][0]
cbs_avg = mean_news_df2[mean_news_df.index=="@cbs"].values[0][0]
cnn_avg = mean_news_df2[mean_news_df.index=="@cnn"].values[0][0]
foxnews_avg = mean_news_df2[mean_news_df.index=="@foxnews"].values[0][0]
nytimes_avg = mean_news_df2[mean_news_df.index=="@nytimes"].values[0][0]
```


```python
plot_news_df = news_df[["compound", "news_source", "tweets_ago"]]

bbc_list = plot_news_df[plot_news_df["news_source"]=="@bbc"]
cbs_list = plot_news_df[plot_news_df["news_source"]=="@cbs"]
cnn_list = plot_news_df[plot_news_df["news_source"]=="@cnn"]
foxnews_list = plot_news_df[plot_news_df["news_source"]=="@foxnews"]
nytimes_list = plot_news_df[plot_news_df["news_source"]=="@nytimes"]

mean_news_df2

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
    </tr>
    <tr>
      <th>news_source</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@bbc</th>
      <td>0.001110</td>
    </tr>
    <tr>
      <th>@cbs</th>
      <td>0.379221</td>
    </tr>
    <tr>
      <th>@cnn</th>
      <td>0.013092</td>
    </tr>
    <tr>
      <th>@foxnews</th>
      <td>-0.016577</td>
    </tr>
    <tr>
      <th>@nytimes</th>
      <td>-0.076053</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_axis = tweets_ago


bbc = plt.scatter(bbc_list['tweets_ago'], bbc_list['compound'], marker="o", facecolors="red", edgecolors="black", alpha=0.75, label ="BBC")
cbs = plt.scatter(cbs_list['tweets_ago'], cbs_list['compound'], marker="o", facecolors="blue", edgecolors="black", alpha=0.75, label = "CBS")
cnn = plt.scatter(cnn_list['tweets_ago'], cnn_list['compound'], marker="o", facecolors="green", edgecolors="black", alpha=0.75, label = "CNN")
foxnews = plt.scatter(foxnews_list['tweets_ago'], foxnews_list['compound'], marker="o", facecolors="yellow", edgecolors="black", alpha=0.75, label = "Fox News")
nytimes = plt.scatter(nytimes_list['tweets_ago'], nytimes_list['compound'], marker="o", facecolors="orange", edgecolors="black", alpha=0.75, label = " NY Times")

plt.legend(bbox_to_anchor = [1,1], handles = [bbc, cbs, cnn, foxnews, nytimes])
plt.gca().invert_xaxis()
sns.set(style="darkgrid")
plt.title("Sentiment Analysis of Media Tweets")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.savefig('output/detailed_media_sentiment.png')
plt.show()
```


![png](output_8_0.png)



```python
# plt.bar(mean_news_df2.index,mean_news_df2["compound"], color = ["red","blue","green","orange","yellow"])
# plt.subplots().set_yticklabels(mean_news_df2["compound"])
# plt.show()

N = 5

fig, ax = plt.subplots(figsize=(10,10))

ind = np.arange(N)


rects1 = ax.bar(ind[0], bbc_avg, color='red', edgecolor='black', alpha=0.75, label='BBC World News')
rects2 = ax.bar(ind[1], cbs_avg, color='yellow', edgecolor='black', alpha=0.75,label='CBS News')
rects3 = ax.bar(ind[2], cnn_avg, color='orange', edgecolor='black', alpha=0.75, label='CNN')
rects4 = ax.bar(ind[3], foxnews_avg, color='blue', edgecolor='black', alpha=0.75, label='Fox News')
rects5 = ax.bar(ind[4], nytimes_avg, color='green', edgecolor='black', alpha=0.75,label='NY Times')

#labels and legends
bar_title = 'Overall Media Sentiment Based on Twitter'
ax.set_title(bar_title, fontsize = 18)
ax.set_xlabel("News Organization", fontsize = 16)
ax.set_ylabel("Overall Tweet Polarity", fontsize = 16)
ax.tick_params(labelsize = 'large')
ax.set_xticklabels(('G1', 'BBC World News', 'CBS News', 'CNN', 'Fox News', 'NY Times'))

plt.savefig('output/overall_media_sentiment.png')

plt.show()
```


![png](output_9_0.png)

